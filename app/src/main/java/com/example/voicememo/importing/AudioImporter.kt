package com.example.voicememo.importing

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.MemoRepository
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.IOException
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Copies a picker-provided audio document into Voice Memo's private storage.
 *
 * The Uri is consumed immediately, so callers do not need a persistable Uri
 * grant. The system picker grants read access for this operation and the app
 * does not need a storage permission.
 */
interface AudioImporter {
    suspend fun importAudio(
        uri: Uri,
        onProgress: (AudioImportProgress) -> Unit = {},
    ): MemoRecord
}

enum class AudioImportPhase {
    PREPARING,
    DECODING,
    SAVING,
    COMPLETE,
}

data class AudioImportProgress(
    val phase: AudioImportPhase,
    /** A best-effort value in 0..1, or null when the provider has no duration. */
    val fraction: Float?,
)

class AudioImportException(
    message: String,
    cause: Throwable? = null,
) : IOException(message, cause)

class AndroidAudioImporter private constructor(
    context: Context,
    private val repository: MemoRepository,
    private val decoder: ContentAudioDecoder,
    private val memoRoot: File,
    private val sourceCacheRoot: File,
    private val clockMillis: () -> Long,
) : AudioImporter {
    private val resolver = context.applicationContext.contentResolver

    constructor(
        context: Context,
        memoRepository: MemoRepository,
    ) : this(
        context = context.applicationContext,
        repository = memoRepository,
        decoder = MediaCodecContentAudioDecoder(),
        memoRoot = File(context.applicationContext.filesDir, "memos"),
        sourceCacheRoot = File(context.applicationContext.cacheDir, "audio-import-sources"),
        clockMillis = System::currentTimeMillis,
    )

    override suspend fun importAudio(
        uri: Uri,
        onProgress: (AudioImportProgress) -> Unit,
    ): MemoRecord = IMPORT_MUTEX.withLock {
        withContext(Dispatchers.IO) {
            var privateSource: File? = null
            try {
                onProgress(AudioImportProgress(AudioImportPhase.PREPARING, 0f))
                cleanStaleImportFiles()
                val metadata = resolveMetadata(uri)
                privateSource = copyToPrivateSource(
                    uri = uri,
                    declaredSize = metadata.sizeBytes,
                    onProgress = onProgress,
                )
                val transaction = AudioImportTransaction(
                    memoRoot = memoRoot,
                    createMemo = repository::createMemo,
                )
                val memo = transaction.import(
                    title = metadata.displayName,
                    createdAtMillis = clockMillis(),
                ) { sink ->
                    decoder.decode(
                        source = requireNotNull(privateSource),
                        sink = sink,
                        onProgress = { fraction ->
                            onProgress(
                                AudioImportProgress(
                                    phase = AudioImportPhase.DECODING,
                                    fraction = fraction,
                                ),
                            )
                        },
                    )
                    onProgress(AudioImportProgress(AudioImportPhase.SAVING, 1f))
                }
                onProgress(AudioImportProgress(AudioImportPhase.COMPLETE, 1f))
                memo
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: AudioImportException) {
                throw failure
            } catch (failure: SecurityException) {
                throw AudioImportException(
                    "The selected audio file can no longer be read. Select it again and retry.",
                    failure,
                )
            } catch (failure: Throwable) {
                throw AudioImportException(
                    failure.message ?: "The selected audio file could not be imported.",
                    failure,
                )
            } finally {
                privateSource?.let { runCatching { it.delete() } }
            }
        }
    }

    private fun resolveMetadata(uri: Uri): SourceMetadata {
        var displayName: String? = null
        var sizeBytes: Long? = null
        runCatching {
            resolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
                null,
                null,
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        .takeIf { it >= 0 && !cursor.isNull(it) }
                        ?.let { displayName = cursor.getString(it) }
                    cursor.getColumnIndex(OpenableColumns.SIZE)
                        .takeIf { it >= 0 && !cursor.isNull(it) }
                        ?.let { sizeBytes = cursor.getLong(it).takeIf { size -> size >= 0L } }
                }
            }
        }

        return SourceMetadata(
            displayName = titleFromDisplayName(displayName),
            sizeBytes = sizeBytes,
        )
    }

    private suspend fun copyToPrivateSource(
        uri: Uri,
        declaredSize: Long?,
        onProgress: (AudioImportProgress) -> Unit,
    ): File {
        check(sourceCacheRoot.mkdirs() || sourceCacheRoot.isDirectory) {
            "A private import directory could not be created."
        }
        val usableBudget = (sourceCacheRoot.usableSpace - FREE_SPACE_RESERVE_BYTES)
            .coerceAtLeast(0L)
        require(usableBudget > 0L && (declaredSize == null || declaredSize <= usableBudget)) {
            "There is not enough free space to import this audio file."
        }
        val target = File(sourceCacheRoot, ".source-${UUID.randomUUID()}")
        val input = resolver.openInputStream(uri)
            ?: throw AudioImportException("The selected audio file could not be opened.")
        try {
            BufferedInputStream(input).use { source ->
                BufferedOutputStream(target.outputStream()).use { output ->
                    val buffer = ByteArray(COPY_BUFFER_BYTES)
                    var copied = 0L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val count = source.read(buffer)
                        if (count < 0) break
                        if (count == 0) continue
                        copied += count
                        require(copied <= usableBudget) {
                            "There is not enough free space to import this audio file."
                        }
                        output.write(buffer, 0, count)
                        val fraction = declaredSize
                            ?.takeIf { it > 0L }
                            ?.let { size -> (copied.toDouble() / size).toFloat().coerceIn(0f, 1f) }
                        onProgress(AudioImportProgress(AudioImportPhase.PREPARING, fraction))
                    }
                }
            }
            require(target.length() > 0L) { "The selected audio file is empty." }
            return target
        } catch (failure: Throwable) {
            runCatching { target.delete() }.exceptionOrNull()?.let(failure::addSuppressed)
            throw failure
        }
    }

    private fun cleanStaleImportFiles() {
        memoRoot.listFiles()
            ?.filter { it.name.startsWith(".importing-") }
            ?.forEach { stale -> runCatching { stale.deleteRecursively() } }
        sourceCacheRoot.listFiles()
            ?.filter { it.name.startsWith(".source-") }
            ?.forEach { stale -> runCatching { stale.delete() } }
    }

    private data class SourceMetadata(
        val displayName: String,
        val sizeBytes: Long?,
    )

    private companion object {
        val IMPORT_MUTEX = Mutex()
        const val COPY_BUFFER_BYTES = 64 * 1024
        const val FREE_SPACE_RESERVE_BYTES = 64L * 1024L * 1024L
    }
}

internal fun titleFromDisplayName(displayName: String?): String {
    val trimmed = TITLE_SEPARATOR_PATTERN
        .replace(displayName.orEmpty(), " ")
        .trim()
    if (trimmed.isEmpty()) return "Imported voice memo"
    val lastDot = trimmed.lastIndexOf('.')
    val withoutExtension = if (lastDot > 0 && lastDot < trimmed.lastIndex) {
        trimmed.substring(0, lastDot)
    } else {
        trimmed
    }
    return withoutExtension
        .trim()
        .takeIf(String::isNotEmpty)
        ?.take(MAX_TITLE_CHARACTERS)
        ?: "Imported voice memo"
}

private val TITLE_SEPARATOR_PATTERN = Regex("[\\p{Cc}\\p{Cf}\\p{Zs}\\s]+")
private const val MAX_TITLE_CHARACTERS = 240
