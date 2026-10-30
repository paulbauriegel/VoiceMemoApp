package com.example.voicememo.backup

import androidx.room.withTransaction
import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.MemoSearchFts
import com.example.voicememo.data.ProjectIcon
import com.example.voicememo.data.ProjectRecord
import com.example.voicememo.data.TranscriptVersion
import com.example.voicememo.data.TranscriptVersionArchive
import com.example.voicememo.data.TranscriptRecord
import com.example.voicememo.data.VoiceMemoDatabase
import com.example.voicememo.data.WaveformState
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.security.MessageDigest
import java.util.Properties
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

enum class BackupPhase {
    READING_LIBRARY,
    WRITING_ARCHIVE,
    READING_ARCHIVE,
    VALIDATING,
    RESTORING,
    COMPLETE,
}

data class BackupProgress(
    val phase: BackupPhase,
    val completedMemos: Int,
    val totalMemos: Int?,
    val currentTitle: String? = null,
)

data class BackupExportResult(
    val memoCount: Int,
)

data class BackupImportResult(
    val importedMemoIds: List<String>,
    val remappedMemoCount: Int,
)

class VoiceMemoBackupException(
    message: String,
    cause: Throwable? = null,
) : IOException(message, cause)

interface VoiceMemoBackupManager {
    suspend fun exportBackup(
        output: OutputStream,
        onProgress: (BackupProgress) -> Unit = {},
    ): BackupExportResult

    suspend fun importBackup(
        input: InputStream,
        onProgress: (BackupProgress) -> Unit = {},
    ): BackupImportResult
}

/**
 * Production backup manager. Streams are supplied by Android's Storage Access
 * Framework; this class never owns or persists a document Uri.
 */
class RoomVoiceMemoBackupManager(
    database: VoiceMemoDatabase,
    memoRoot: File,
) : VoiceMemoBackupManager by DefaultVoiceMemoBackupManager(
    store = RoomVoiceMemoBackupStore(database),
    memoRoot = memoRoot,
)

internal data class BackupMemoSnapshot(
    val memo: MemoRecord,
    val transcript: TranscriptRecord?,
    val project: ProjectRecord? = null,
    val versions: List<TranscriptVersion> = emptyList(),
)

internal data class RestoredMemo(
    val memo: MemoRecord,
    val transcript: TranscriptRecord?,
    val project: ProjectRecord? = null,
    val versions: List<TranscriptVersion> = emptyList(),
)

internal interface VoiceMemoBackupStore {
    suspend fun snapshots(): List<BackupMemoSnapshot>

    suspend fun everyMemoId(): Set<String>

    suspend fun insertRestored(memos: List<RestoredMemo>)
}

private class RoomVoiceMemoBackupStore(
    private val database: VoiceMemoDatabase,
) : VoiceMemoBackupStore {
    private val dao = database.voiceMemoDao()

    override suspend fun snapshots(): List<BackupMemoSnapshot> =
        database.withTransaction {
            val projects = dao.getAllProjects().associateBy(ProjectRecord::id)
            dao.getAllMemoAggregates().map { aggregate ->
                BackupMemoSnapshot(
                    memo = aggregate.memo,
                    transcript = aggregate.transcripts.singleOrNull(),
                    project = aggregate.memo.projectId?.let(projects::get),
                    versions = dao.getTranscriptVersions(aggregate.memo.id).map { version ->
                        if (version.selected) version.copy(
                            transcript = aggregate.transcripts.singleOrNull() ?: version.transcript,
                        ) else version
                    },
                )
            }
        }

    override suspend fun everyMemoId(): Set<String> =
        database.withTransaction { dao.getEveryMemoIds().toSet() }

    override suspend fun insertRestored(memos: List<RestoredMemo>) {
        database.withTransaction {
            memos.mapNotNull(RestoredMemo::project)
                .distinctBy(ProjectRecord::id)
                .forEach { dao.insertProjectIfAbsent(it) }
            memos.forEach { restored ->
                dao.insertMemo(restored.memo)
                restored.transcript?.let { dao.upsertTranscript(it) }
                restored.versions.forEach { dao.upsertTranscriptVersion(it) }
                dao.insertSearchEntry(
                    MemoSearchFts(
                        memoId = restored.memo.id,
                        title = restored.memo.title,
                        effectiveTranscript = restored.transcript?.effectiveText.orEmpty(),
                    ),
                )
            }
        }
    }
}

internal class DefaultVoiceMemoBackupManager(
    private val store: VoiceMemoBackupStore,
    private val memoRoot: File,
    private val clockMillis: () -> Long = System::currentTimeMillis,
    private val newId: () -> String = { UUID.randomUUID().toString() },
    private val usableSpace: () -> Long = { memoRoot.usableSpace },
) : VoiceMemoBackupManager {
    private val operationMutex = Mutex()

    override suspend fun exportBackup(
        output: OutputStream,
        onProgress: (BackupProgress) -> Unit,
    ): BackupExportResult = operationMutex.withLock {
        withContext(Dispatchers.IO) {
            try {
                val root = canonicalMemoRoot()
                reconcilePendingRestoreDirectories(root)
                cleanStaleRestoreDirectories(root)
                onProgress(BackupProgress(BackupPhase.READING_LIBRARY, 0, null))
                val snapshots = store.snapshots()
                require(snapshots.size <= MAX_MEMOS) {
                    "This library contains too many memos for one backup."
                }
                val prepared = snapshots.map(::prepareExportMemo)
                require(prepared.sumOf { memo ->
                    (memo.generatedText?.size?.toLong() ?: 0L) +
                        (memo.editedText?.size?.toLong() ?: 0L) +
                        (memo.effectiveText?.size?.toLong() ?: 0L) +
                        (memo.versions?.size?.toLong() ?: 0L)
                } <= MAX_TOTAL_TRANSCRIPT_BYTES) { "The backup contains too much transcript text." }
                onProgress(BackupProgress(BackupPhase.WRITING_ARCHIVE, 0, prepared.size))

                val zip = ZipOutputStream(BufferedOutputStream(output))
                writeZipBytes(
                    zip = zip,
                    name = MANIFEST_ENTRY,
                    bytes = manifestProperties(prepared).toBytes(),
                )
                prepared.forEachIndexed { index, memo ->
                    currentCoroutineContext().ensureActive()
                    val base = "$MEMOS_DIRECTORY/${memo.snapshot.memo.id}"
                    writeZipBytes(
                        zip,
                        "$base/$METADATA_FILE",
                        memo.metadata.toBytes(),
                    )
                    writeZipFile(zip, "$base/$AUDIO_FILE", memo.audio)
                    memo.waveform?.let { writeZipFile(zip, "$base/$WAVEFORM_FILE", it) }
                    memo.generatedText?.let {
                        writeZipBytes(zip, "$base/$GENERATED_TRANSCRIPT_FILE", it)
                    }
                    memo.editedText?.let {
                        writeZipBytes(zip, "$base/$EDITED_TRANSCRIPT_FILE", it)
                    }
                    memo.versions?.let {
                        writeZipBytes(zip, "$base/$VERSIONS_FILE", it)
                    }
                    memo.effectiveText?.let {
                        writeZipBytes(zip, "$base/$EFFECTIVE_TRANSCRIPT_FILE", it)
                    }
                    onProgress(
                        BackupProgress(
                            phase = BackupPhase.WRITING_ARCHIVE,
                            completedMemos = index + 1,
                            totalMemos = prepared.size,
                            currentTitle = memo.snapshot.memo.title,
                        ),
                    )
                }
                zip.finish()
                zip.flush()
                onProgress(
                    BackupProgress(
                        BackupPhase.COMPLETE,
                        prepared.size,
                        prepared.size,
                    ),
                )
                BackupExportResult(memoCount = prepared.size)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: VoiceMemoBackupException) {
                throw failure
            } catch (failure: Throwable) {
                throw VoiceMemoBackupException(
                    failure.message ?: "The backup could not be created.",
                    failure,
                )
            }
        }
    }

    override suspend fun importBackup(
        input: InputStream,
        onProgress: (BackupProgress) -> Unit,
    ): BackupImportResult = operationMutex.withLock {
        withContext(Dispatchers.IO) {
            val root = canonicalMemoRoot()
            reconcilePendingRestoreDirectories(root)
            cleanStaleRestoreDirectories(root)
            val staging = File(root, ".restoring-${UUID.randomUUID()}")
            require(staging.mkdirs()) { "A private restore directory could not be created." }
            val published = mutableListOf<File>()
            val readyDirectories = mutableListOf<File>()
            var committed = false
            try {
                onProgress(BackupProgress(BackupPhase.READING_ARCHIVE, 0, null))
                extractArchive(input, staging)
                onProgress(BackupProgress(BackupPhase.VALIDATING, 0, null))
                val parsed = parseStagedArchive(staging)
                val existingIds = store.everyMemoId().toMutableSet()
                val restored = mutableListOf<RestoredMemo>()
                var remapped = 0

                parsed.forEachIndexed { index, source ->
                    currentCoroutineContext().ensureActive()
                    val destinationId = chooseDestinationId(
                        originalId = source.originalId,
                        occupiedIds = existingIds,
                        root = root,
                    ).also {
                        if (it != source.originalId) remapped += 1
                        existingIds += it
                    }
                    val ready = File(root, ".restore-ready-$destinationId-${UUID.randomUUID()}")
                    check(ready.mkdirs()) { "A restored memo could not be staged." }
                    readyDirectories += ready
                    File(ready, PENDING_RESTORE_MARKER).writeText(
                        source.originalId,
                        Charsets.US_ASCII,
                    )
                    val destinationAudio = File(ready, AUDIO_FILE)
                    moveWithinMemoRoot(source.audio, destinationAudio)
                    val destinationWaveform = source.waveform?.let { waveform ->
                        File(ready, WAVEFORM_FILE).also {
                            moveWithinMemoRoot(waveform, it)
                        }
                    }
                    val finalDirectory = File(root, destinationId)
                    check(!finalDirectory.exists()) {
                        "A memo directory already exists for $destinationId."
                    }
                    check(ready.renameTo(finalDirectory)) {
                        "A restored memo could not be published."
                    }
                    readyDirectories -= ready
                    published += finalDirectory

                    val finalAudio = File(finalDirectory, AUDIO_FILE)
                    val finalWaveform = destinationWaveform?.let {
                        File(finalDirectory, WAVEFORM_FILE)
                    }
                    val memo = MemoRecord(
                        id = destinationId,
                        title = source.title,
                        createdAtMillis = source.createdAtMillis,
                        updatedAtMillis = source.updatedAtMillis,
                        durationMillis = source.durationMillis,
                        audioPath = finalAudio.absolutePath,
                        waveformPath = finalWaveform?.absolutePath,
                        waveformState = source.waveformState,
                        projectId = source.project?.id,
                        languageOverride = source.languageOverride,
                        pendingDeletion = false,
                        isStarred = source.isStarred,
                        titleEditedByUser = source.titleEditedByUser,
                        generatedSummary = source.generatedSummary,
                        metadataGeneratedAtMillis = source.metadataGeneratedAtMillis,
                        metadataModelId = source.metadataModelId,
                    )
                    val transcript = source.transcript?.copy(memoId = destinationId)
                    restored += RestoredMemo(
                        memo = memo,
                        transcript = transcript,
                        project = source.project,
                        versions = source.versions.map { version -> version.copy(
                            memoId = destinationId,
                            transcript = version.transcript.copy(memoId = destinationId),
                        ) },
                    )
                    onProgress(
                        BackupProgress(
                            phase = BackupPhase.RESTORING,
                            completedMemos = index + 1,
                            totalMemos = parsed.size,
                            currentTitle = source.title,
                        ),
                    )
                }

                withContext(NonCancellable) {
                    store.insertRestored(restored)
                    committed = true
                    published.forEach { directory ->
                        runCatching { File(directory, PENDING_RESTORE_MARKER).delete() }
                    }
                }
                runCatching { staging.deleteRecursively() }
                onProgress(
                    BackupProgress(
                        BackupPhase.COMPLETE,
                        restored.size,
                        restored.size,
                    ),
                )
                BackupImportResult(
                    importedMemoIds = restored.map { it.memo.id },
                    remappedMemoCount = remapped,
                )
            } catch (cancelled: CancellationException) {
                if (!committed) {
                    readyDirectories.forEach(::deleteRestoreDirectory)
                    published.forEach(::deleteRestoreDirectory)
                }
                runCatching { staging.deleteRecursively() }
                throw cancelled
            } catch (failure: Throwable) {
                if (!committed) {
                    readyDirectories.forEach(::deleteRestoreDirectory)
                    published.forEach(::deleteRestoreDirectory)
                }
                runCatching { staging.deleteRecursively() }
                throw if (failure is VoiceMemoBackupException) {
                    failure
                } else {
                    VoiceMemoBackupException(
                        failure.message ?: "The selected file is not a valid Voice Memo backup.",
                        failure,
                    )
                }
            }
        }
    }

    private fun prepareExportMemo(snapshot: BackupMemoSnapshot): PreparedExportMemo {
        val memo = snapshot.memo
        requireValidUuid(memo.id)
        require(memo.title.isNotBlank() && memo.title.length <= MAX_TITLE_CHARACTERS) {
            "A memo has an invalid title."
        }
        require(memo.durationMillis >= 0L) { "A memo has an invalid duration." }
        val audio = privateMemoFile(memo.audioPath)
        require(audio.isFile && audio.length() > 0L) {
            "Audio for “${memo.title}” is missing."
        }
        require(audio.length() <= MAX_AUDIO_BYTES) {
            "Audio for “${memo.title}” is too large to back up."
        }
        validatePcm16MonoWav(audio, memo.durationMillis)
        val waveform = memo.waveformPath
            ?.let(::privateMemoFile)
            ?.takeIf(File::isFile)
        require(memo.waveformState != WaveformState.READY || waveform != null) {
            "Waveform data for “${memo.title}” is missing."
        }
        require(waveform == null || waveform.length() <= MAX_WAVEFORM_BYTES) {
            "Waveform data for “${memo.title}” is too large."
        }
        waveform?.let { validateWaveformSidecar(it, memo.durationMillis) }

        val transcript = snapshot.transcript
        require(transcript == null || transcript.memoId == memo.id) {
            "A transcript does not belong to its memo."
        }
        val generated = transcript?.generatedText?.toUtf8Bounded()
        val edited = transcript?.editedText?.toUtf8Bounded()
        val effective = transcript?.effectiveText?.toUtf8Bounded()
        val versions = snapshot.versions.takeIf { it.isNotEmpty() }
            ?.let(TranscriptVersionArchive::encode)
        require(versions == null || versions.size <= MAX_TRANSCRIPT_BYTES) {
            "Transcript history is too large to back up."
        }
        val metadata = Properties().apply {
            setProperty("id", memo.id)
            setProperty("title", memo.title)
            setProperty("createdAtMillis", memo.createdAtMillis.toString())
            setProperty("updatedAtMillis", memo.updatedAtMillis.toString())
            setProperty("durationMillis", memo.durationMillis.toString())
            setProperty("waveformState", memo.waveformState.name)
            setProperty("isStarred", memo.isStarred.toString())
            setProperty("titleEditedByUser", memo.titleEditedByUser.toString())
            putOptionalString("generatedSummary", memo.generatedSummary)
            putOptionalLong("metadataGeneratedAtMillis", memo.metadataGeneratedAtMillis)
            putOptionalString("metadataModelId", memo.metadataModelId)
            putOptionalString("languageOverride", memo.languageOverride)
            putOptionalString("projectId", snapshot.project?.id)
            putOptionalString("projectName", snapshot.project?.name)
            putOptionalLong("projectCreatedAtMillis", snapshot.project?.createdAtMillis)
            putOptionalLong("projectUpdatedAtMillis", snapshot.project?.updatedAtMillis)
            putOptionalString("projectIcon", snapshot.project?.icon?.name)
            snapshot.project?.let {
                setProperty("projectArchived", it.isArchived.toString())
            }
            setProperty("audio.bytes", audio.length().toString())
            setProperty("audio.sha256", sha256(audio))
            setProperty("waveform.present", (waveform != null).toString())
            waveform?.let {
                setProperty("waveform.bytes", it.length().toString())
                setProperty("waveform.sha256", sha256(it))
            }
            versions?.let {
                setProperty("versions.bytes", it.size.toString())
                setProperty("versions.sha256", sha256(it))
            }
            setProperty("transcript.present", (transcript != null).toString())
            setProperty("generated.present", (generated != null).toString())
            generated?.let {
                setProperty("generated.bytes", it.size.toString())
                setProperty("generated.sha256", sha256(it))
            }
            setProperty("edited.present", (edited != null).toString())
            edited?.let {
                setProperty("edited.bytes", it.size.toString())
                setProperty("edited.sha256", sha256(it))
            }
            setProperty("effective.present", (effective != null).toString())
            effective?.let {
                setProperty("effective.bytes", it.size.toString())
                setProperty("effective.sha256", sha256(it))
            }
            putOptionalString("detectedLanguage", transcript?.detectedLanguage)
            putOptionalLong("generatedAtMillis", transcript?.generatedAtMillis)
            putOptionalLong("editedAtMillis", transcript?.editedAtMillis)
            setProperty(
                "generatedRevision",
                (transcript?.generatedRevision ?: 0L).toString(),
            )
        }
        return PreparedExportMemo(
            snapshot = snapshot,
            audio = audio,
            waveform = waveform,
            generatedText = generated,
            editedText = edited,
            effectiveText = effective,
            versions = versions,
            metadata = metadata,
        )
    }

    private suspend fun extractArchive(input: InputStream, staging: File) {
        val availableSpace = usableSpace().coerceAtLeast(0L)
        val reserve = maxOf(
            MIN_FREE_SPACE_RESERVE_BYTES,
            availableSpace / FREE_SPACE_RESERVE_DIVISOR,
        )
        val freeBudget = (availableSpace - reserve).coerceAtLeast(0L)
        val totalBudget = minOf(MAX_ARCHIVE_UNCOMPRESSED_BYTES, freeBudget)
        require(totalBudget > 0L) { "There is not enough free space to restore this backup." }
        val seen = hashSetOf<String>()
        val zip = ZipInputStream(BufferedInputStream(input))
        var totalBytes = 0L
        var totalTranscriptBytes = 0L
        var entryCount = 0
        while (true) {
            currentCoroutineContext().ensureActive()
            val entry = zip.nextEntry ?: break
            entryCount += 1
            require(entryCount <= MAX_ENTRIES) { "The backup contains too many files." }
            val name = entry.name
            require(!entry.isDirectory && isAllowedEntryName(name)) {
                "The backup contains an unsafe or unknown file: $name"
            }
            require(seen.add(name)) { "The backup contains a duplicate file: $name" }
            val entryLimit = maximumEntryBytes(name)
            if (entry.size >= 0L) {
                require(entry.size <= entryLimit && entry.size <= totalBudget - totalBytes) {
                    "A backup entry is too large: $name"
                }
            }
            val target = safeStagingFile(staging, name)
            val parent = requireNotNull(target.parentFile)
            check(parent.mkdirs() || parent.isDirectory) {
                "A restore directory could not be created."
            }
            BufferedOutputStream(target.outputStream()).use { output ->
                val buffer = ByteArray(COPY_BUFFER_BYTES)
                var entryBytes = 0L
                while (true) {
                    currentCoroutineContext().ensureActive()
                    val count = zip.read(buffer)
                    if (count < 0) break
                    if (count == 0) continue
                    entryBytes += count
                    require(entryBytes <= entryLimit) { "A backup entry is too large: $name" }
                    require(count.toLong() <= totalBudget - totalBytes) {
                        "The backup is too large for this device."
                    }
                    if (isTranscriptEntry(name)) {
                        require(count.toLong() <= MAX_TOTAL_TRANSCRIPT_BYTES - totalTranscriptBytes) {
                            "The backup contains too much transcript text."
                        }
                        totalTranscriptBytes += count
                    }
                    totalBytes += count
                    output.write(buffer, 0, count)
                }
            }
            zip.closeEntry()
        }
        require(seen.isNotEmpty()) { "The selected backup is empty." }
    }

    private fun parseStagedArchive(staging: File): List<ParsedMemo> {
        val manifestFile = File(staging, MANIFEST_ENTRY)
        val manifest = loadProperties(manifestFile, MAX_MANIFEST_BYTES)
        require(manifest.required("format") == FORMAT_ID) {
            "This is not a Voice Memo backup."
        }
        require(manifest.requiredInt("version") == FORMAT_VERSION) {
            "This backup version is not supported."
        }
        val memoCount = manifest.requiredInt("memoCount")
        require(memoCount in 0..MAX_MEMOS) { "The backup has an invalid memo count." }
        val ids = List(memoCount) { index ->
            manifest.required("memo.$index").also(::requireValidUuid)
        }
        require(ids.toSet().size == ids.size) { "The backup lists a memo more than once." }

        val actualFolders = File(staging, MEMOS_DIRECTORY)
            .listFiles()
            .orEmpty()
            .filter(File::isDirectory)
            .mapTo(linkedSetOf(), File::getName)
        require(actualFolders == ids.toSet()) {
            "The backup memo list does not match its files."
        }
        return ids.map { id -> parseMemo(File(staging, "$MEMOS_DIRECTORY/$id"), id) }
    }

    private fun parseMemo(directory: File, originalId: String): ParsedMemo {
        val metadata = loadProperties(File(directory, METADATA_FILE), MAX_METADATA_BYTES)
        require(metadata.required("id") == originalId) {
            "Backup metadata does not match its memo directory."
        }
        val title = metadata.required("title")
        require(title.isNotBlank() && title.length <= MAX_TITLE_CHARACTERS) {
            "A backup memo has an invalid title."
        }
        val createdAt = metadata.requiredLong("createdAtMillis")
        val updatedAt = metadata.requiredLong("updatedAtMillis")
        val duration = metadata.requiredLong("durationMillis")
        require(createdAt >= 0L && updatedAt >= 0L && duration >= 0L) {
            "A backup memo has invalid timestamps or duration."
        }
        val waveformState = runCatching {
            WaveformState.valueOf(metadata.required("waveformState"))
        }.getOrElse { throw VoiceMemoBackupException("A backup has an invalid waveform state.", it) }
        val languageOverride = metadata.optionalString("languageOverride")
            ?.also(::requireValidLanguage)
        val isStarred = metadata.optionalBooleanIfDeclared("isStarred") ?: false
        val titleEditedByUser =
            metadata.optionalBooleanIfDeclared("titleEditedByUser") ?: true
        val generatedSummary = metadata.optionalStringIfDeclared("generatedSummary")
            ?.also {
                require(it.isNotBlank() && it.length <= MAX_GENERATED_SUMMARY_CHARACTERS) {
                    "A backup has an invalid generated summary."
                }
            }
        val metadataGeneratedAt =
            metadata.optionalLongIfDeclared("metadataGeneratedAtMillis")
        val metadataModelId = metadata.optionalStringIfDeclared("metadataModelId")
        require(metadataGeneratedAt == null || metadataGeneratedAt >= 0L) {
            "A backup has an invalid metadata timestamp."
        }
        require(metadataModelId == null || metadataModelId.length <= MAX_METADATA_MODEL_ID_CHARACTERS) {
            "A backup has an invalid metadata model ID."
        }
        require(
            (generatedSummary == null && metadataGeneratedAt == null && metadataModelId == null) ||
                (generatedSummary != null && metadataGeneratedAt != null && metadataModelId != null),
        ) {
            "A backup has incomplete generated metadata."
        }
        val projectId = metadata.optionalStringIfDeclared("projectId")
        val projectName = metadata.optionalStringIfDeclared("projectName")
        val projectCreatedAt = metadata.optionalLongIfDeclared("projectCreatedAtMillis")
        val projectUpdatedAt = metadata.optionalLongIfDeclared("projectUpdatedAtMillis")
        val projectIcon = metadata.optionalStringIfDeclared("projectIcon")
            ?.let { value ->
                runCatching { ProjectIcon.valueOf(value) }.getOrElse {
                    throw VoiceMemoBackupException("A backup has an invalid project icon.", it)
                }
            }
            ?: ProjectIcon.WORK
        val projectArchived = metadata.optionalBooleanIfDeclared("projectArchived") ?: false
        require((projectId == null) == (projectName == null)) {
            "A backup has incomplete project metadata."
        }
        require(
            projectId == null ||
                (projectCreatedAt != null && projectUpdatedAt != null),
        ) {
            "A backup has incomplete project timestamps."
        }
        val project = projectId?.let { id ->
            requireValidUuid(id)
            require(projectName!!.isNotBlank() && projectName.length <= MAX_PROJECT_NAME_CHARACTERS) {
                "A backup has an invalid project name."
            }
            val safeCreatedAt = requireNotNull(projectCreatedAt)
            val safeUpdatedAt = requireNotNull(projectUpdatedAt)
            require(safeCreatedAt >= 0L && safeUpdatedAt >= safeCreatedAt) {
                "A backup has invalid project timestamps."
            }
            ProjectRecord(
                id = id,
                name = projectName,
                createdAtMillis = safeCreatedAt,
                updatedAtMillis = safeUpdatedAt,
                icon = projectIcon,
                isArchived = projectArchived,
            )
        }
        val audio = File(directory, AUDIO_FILE)
        verifyFile(
            file = audio,
            expectedBytes = metadata.requiredLong("audio.bytes"),
            expectedSha256 = metadata.requiredSha256("audio.sha256"),
            maximumBytes = MAX_AUDIO_BYTES,
            label = "audio",
        )
        require(audio.length() > 0L) { "A backup contains empty audio." }
        validatePcm16MonoWav(audio, duration)

        val waveformPresent = metadata.requiredBoolean("waveform.present")
        val waveform = File(directory, WAVEFORM_FILE).takeIf(File::isFile)
        require(waveformPresent == (waveform != null)) {
            "A backup has inconsistent waveform metadata."
        }
        waveform?.let {
            verifyFile(
                file = it,
                expectedBytes = metadata.requiredLong("waveform.bytes"),
                expectedSha256 = metadata.requiredSha256("waveform.sha256"),
                maximumBytes = MAX_WAVEFORM_BYTES,
                label = "waveform",
            )
            validateWaveformSidecar(it, duration)
        }
        require(waveformState != WaveformState.READY || waveform != null) {
            "A ready waveform is missing from the backup."
        }

        val transcriptPresent = metadata.requiredBoolean("transcript.present")
        val generated = readOptionalText(
            directory,
            GENERATED_TRANSCRIPT_FILE,
            metadata,
            "generated",
        )
        val edited = readOptionalText(
            directory,
            EDITED_TRANSCRIPT_FILE,
            metadata,
            "edited",
        )
        val effective = readOptionalText(
            directory,
            EFFECTIVE_TRANSCRIPT_FILE,
            metadata,
            "effective",
        )
        require(
            transcriptPresent == (
                generated != null ||
                    edited != null ||
                    metadata.requiredBoolean("effective.present")
                ),
        ) {
            "A backup has inconsistent transcript metadata."
        }
        val expectedEffective = if (transcriptPresent) edited ?: generated.orEmpty() else null
        require(effective == expectedEffective) {
            "The readable transcript does not match the backed-up transcript."
        }
        val detectedLanguage = metadata.optionalString("detectedLanguage")
            ?.also(::requireValidLanguage)
        val generatedAt = metadata.optionalLong("generatedAtMillis")
        val editedAt = metadata.optionalLong("editedAtMillis")
        require(generatedAt == null || generatedAt >= 0L) {
            "A backup has an invalid transcript timestamp."
        }
        require(editedAt == null || editedAt >= 0L) {
            "A backup has an invalid transcript timestamp."
        }
        val revision = metadata.requiredLong("generatedRevision")
        require(revision >= 0L) { "A backup has an invalid transcript revision." }
        val transcript = if (transcriptPresent) {
            TranscriptRecord(
                memoId = originalId,
                generatedText = generated,
                editedText = edited,
                detectedLanguage = detectedLanguage,
                generatedAtMillis = generatedAt,
                editedAtMillis = editedAt,
                generatedRevision = revision,
            )
        } else {
            require(generated == null && edited == null && effective == null) {
                "A backup contains unexpected transcript files."
            }
            null
        }

        val versionsFile = File(directory, VERSIONS_FILE)
        val versions = if (metadata.containsKey("versions.bytes")) {
            verifyFile(versionsFile, metadata.requiredLong("versions.bytes"),
                metadata.requiredSha256("versions.sha256"), MAX_TRANSCRIPT_BYTES, "transcript history")
            TranscriptVersionArchive.decode(originalId, versionsFile.readBytes()).also { history ->
                require(history.firstOrNull { it.selected }?.transcript == transcript) {
                    "The selected transcript does not match its history."
                }
            }
        } else {
            require(!versionsFile.exists()) { "Unexpected transcript history." }
            emptyList()
        }

        return ParsedMemo(
            originalId = originalId,
            title = title,
            createdAtMillis = createdAt,
            updatedAtMillis = updatedAt,
            durationMillis = duration,
            waveformState = waveformState,
            languageOverride = languageOverride,
            isStarred = isStarred,
            titleEditedByUser = titleEditedByUser,
            generatedSummary = generatedSummary,
            metadataGeneratedAtMillis = metadataGeneratedAt,
            metadataModelId = metadataModelId,
            project = project,
            audio = audio,
            waveform = waveform,
            transcript = transcript,
            versions = versions,
        )
    }

    private fun readOptionalText(
        directory: File,
        fileName: String,
        metadata: Properties,
        key: String,
    ): String? {
        val present = metadata.requiredBoolean("$key.present")
        val file = File(directory, fileName)
        require(present == file.isFile) { "A backup has inconsistent $key transcript metadata." }
        if (!present) return null
        verifyFile(
            file = file,
            expectedBytes = metadata.requiredLong("$key.bytes"),
            expectedSha256 = metadata.requiredSha256("$key.sha256"),
            maximumBytes = MAX_TRANSCRIPT_BYTES,
            label = "$key transcript",
        )
        return decodeUtf8(file.readBytes())
    }

    private fun chooseDestinationId(
        originalId: String,
        occupiedIds: Set<String>,
        root: File,
    ): String {
        if (originalId !in occupiedIds && !File(root, originalId).exists()) return originalId
        repeat(MAX_ID_ATTEMPTS) {
            val candidate = newId()
            requireValidUuid(candidate)
            if (candidate !in occupiedIds && !File(root, candidate).exists()) return candidate
        }
        error("A unique memo identifier could not be created.")
    }

    private fun manifestProperties(memos: List<PreparedExportMemo>): Properties =
        Properties().apply {
            setProperty("format", FORMAT_ID)
            setProperty("version", FORMAT_VERSION.toString())
            setProperty("createdAtMillis", clockMillis().toString())
            setProperty("memoCount", memos.size.toString())
            memos.forEachIndexed { index, memo ->
                setProperty("memo.$index", memo.snapshot.memo.id)
            }
        }

    private fun canonicalMemoRoot(): File {
        check(memoRoot.mkdirs() || memoRoot.isDirectory) {
            "The private memo directory is unavailable."
        }
        return memoRoot.canonicalFile
    }

    private fun privateMemoFile(path: String): File {
        val root = canonicalMemoRoot()
        val file = File(path).canonicalFile
        require(file.path.startsWith(root.path + File.separator)) {
            "Refusing to back up a file outside private memo storage."
        }
        return file
    }

    private fun cleanStaleRestoreDirectories(root: File) {
        root.listFiles()
            .orEmpty()
            .filter {
                it.isDirectory &&
                    (it.name.startsWith(".restoring-") || it.name.startsWith(".restore-ready-"))
            }
            .forEach(::deleteRestoreDirectory)
    }

    private suspend fun reconcilePendingRestoreDirectories(root: File) {
        val memoIds = store.everyMemoId()
        root.listFiles()
            .orEmpty()
            .filter { directory ->
                directory.isDirectory &&
                    File(directory, PENDING_RESTORE_MARKER).isFile
            }
            .forEach { directory ->
                if (directory.name in memoIds) {
                    runCatching { File(directory, PENDING_RESTORE_MARKER).delete() }
                } else {
                    deleteRestoreDirectory(directory)
                }
            }
    }

    private fun deleteRestoreDirectory(directory: File) {
        val root = canonicalMemoRoot()
        val canonical = directory.canonicalFile
        require(canonical.parentFile == root) {
            "Refusing to clean a path outside private memo storage."
        }
        canonical.deleteRecursively()
    }

    private fun moveWithinMemoRoot(source: File, destination: File) {
        privateMemoFile(source.absolutePath)
        privateMemoFile(destination.parentFile!!.absolutePath + File.separator + "placeholder")
        check(source.renameTo(destination)) { "A restored file could not be staged." }
    }

    private data class PreparedExportMemo(
        val snapshot: BackupMemoSnapshot,
        val audio: File,
        val waveform: File?,
        val generatedText: ByteArray?,
        val editedText: ByteArray?,
        val effectiveText: ByteArray?,
        val versions: ByteArray?,
        val metadata: Properties,
    )

    private data class ParsedMemo(
        val originalId: String,
        val title: String,
        val createdAtMillis: Long,
        val updatedAtMillis: Long,
        val durationMillis: Long,
        val waveformState: WaveformState,
        val languageOverride: String?,
        val isStarred: Boolean,
        val titleEditedByUser: Boolean,
        val generatedSummary: String?,
        val metadataGeneratedAtMillis: Long?,
        val metadataModelId: String?,
        val project: ProjectRecord?,
        val audio: File,
        val waveform: File?,
        val transcript: TranscriptRecord?,
        val versions: List<TranscriptVersion>,
    )
}

private fun writeZipBytes(zip: ZipOutputStream, name: String, bytes: ByteArray) {
    zip.putNextEntry(ZipEntry(name).apply { time = 0L })
    zip.write(bytes)
    zip.closeEntry()
}

private suspend fun writeZipFile(zip: ZipOutputStream, name: String, file: File) {
    zip.putNextEntry(ZipEntry(name).apply { time = 0L })
    BufferedInputStream(file.inputStream()).use { input ->
        val buffer = ByteArray(COPY_BUFFER_BYTES)
        while (true) {
            currentCoroutineContext().ensureActive()
            val count = input.read(buffer)
            if (count < 0) break
            if (count > 0) zip.write(buffer, 0, count)
        }
    }
    zip.closeEntry()
}

private fun Properties.toBytes(): ByteArray =
    ByteArrayOutputStream().use { output ->
        store(output, null)
        output.toByteArray()
    }

private fun loadProperties(file: File, maximumBytes: Long): Properties {
    require(file.isFile && file.length() <= maximumBytes) {
        "Backup metadata is missing or too large."
    }
    return Properties().apply {
        ByteArrayInputStream(file.readBytes()).use(::load)
    }
}

private fun Properties.required(key: String): String =
    getProperty(key) ?: throw VoiceMemoBackupException("Backup metadata is missing “$key”.")

private fun Properties.requiredInt(key: String): Int =
    required(key).toIntOrNull()
        ?: throw VoiceMemoBackupException("Backup metadata “$key” is invalid.")

private fun Properties.requiredLong(key: String): Long =
    required(key).toLongOrNull()
        ?: throw VoiceMemoBackupException("Backup metadata “$key” is invalid.")

private fun Properties.requiredBoolean(key: String): Boolean =
    when (val value = required(key)) {
        "true" -> true
        "false" -> false
        else -> throw VoiceMemoBackupException("Backup metadata “$key” is invalid: $value")
    }

private fun Properties.optionalBooleanIfDeclared(key: String): Boolean? =
    if (containsKey(key)) requiredBoolean(key) else null

private fun Properties.requiredSha256(key: String): String =
    required(key).lowercase().also {
        require(SHA256_PATTERN.matches(it)) { "Backup metadata “$key” is invalid." }
    }

private fun Properties.putOptionalString(key: String, value: String?) {
    setProperty("$key.present", (value != null).toString())
    value?.let { setProperty(key, it) }
}

private fun Properties.optionalString(key: String): String? =
    if (requiredBoolean("$key.present")) required(key) else null

private fun Properties.optionalStringIfDeclared(key: String): String? =
    if (containsKey("$key.present")) optionalString(key) else null

private fun Properties.putOptionalLong(key: String, value: Long?) {
    setProperty("$key.present", (value != null).toString())
    value?.let { setProperty(key, it.toString()) }
}

private fun Properties.optionalLong(key: String): Long? =
    if (requiredBoolean("$key.present")) requiredLong(key) else null

private fun Properties.optionalLongIfDeclared(key: String): Long? =
    if (containsKey("$key.present")) optionalLong(key) else null

private fun String.toUtf8Bounded(): ByteArray =
    toByteArray(Charsets.UTF_8).also {
        require(it.size.toLong() <= MAX_TRANSCRIPT_BYTES) {
            "A transcript is too large to back up."
        }
    }

private fun decodeUtf8(bytes: ByteArray): String {
    val decoder = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
    return try {
        decoder.decode(ByteBuffer.wrap(bytes)).toString()
    } catch (failure: Throwable) {
        throw VoiceMemoBackupException("A transcript is not valid UTF-8.", failure)
    }
}

private fun validatePcm16MonoWav(
    file: File,
    expectedDurationMillis: Long,
) {
    RandomAccessFile(file, "r").use { input ->
        val fileLength = input.length()
        require(fileLength >= MIN_WAV_BYTES) { "The backed-up audio is not a complete WAV file." }
        require(input.readAscii(4) == "RIFF") { "The backed-up audio is not RIFF WAV." }
        val riffSize = input.readUInt32Le()
        require(riffSize + 8L == fileLength) { "The backed-up WAV has an invalid RIFF size." }
        require(input.readAscii(4) == "WAVE") { "The backed-up audio is not WAVE audio." }

        var formatFound = false
        var dataSize: Long? = null
        var sampleRate = 0L
        while (input.filePointer + 8L <= fileLength) {
            val chunkId = input.readAscii(4)
            val chunkSize = input.readUInt32Le()
            val payloadStart = input.filePointer
            require(chunkSize <= fileLength - payloadStart) {
                "The backed-up WAV contains a truncated chunk."
            }
            val payloadEnd = payloadStart + chunkSize
            when (chunkId) {
                "fmt " -> {
                    require(!formatFound && chunkSize >= 16L) {
                        "The backed-up WAV has an invalid format chunk."
                    }
                    val audioFormat = input.readUInt16Le()
                    val channelCount = input.readUInt16Le()
                    sampleRate = input.readUInt32Le()
                    val byteRate = input.readUInt32Le()
                    val blockAlign = input.readUInt16Le()
                    val bitsPerSample = input.readUInt16Le()
                    require(
                        audioFormat == 1 &&
                            channelCount == 1 &&
                            sampleRate in MIN_AUDIO_SAMPLE_RATE..MAX_AUDIO_SAMPLE_RATE &&
                            bitsPerSample == 16 &&
                            blockAlign == 2 &&
                            byteRate == sampleRate * 2L
                    ) {
                        "The backup audio must be mono PCM16 WAV."
                    }
                    formatFound = true
                }

                "data" -> {
                    require(dataSize == null && chunkSize > 0L && chunkSize % 2L == 0L) {
                        "The backed-up WAV has invalid PCM data."
                    }
                    dataSize = chunkSize
                }
            }
            val nextChunk = payloadEnd + (chunkSize and 1L)
            require(nextChunk <= fileLength) { "The backed-up WAV has invalid chunk padding." }
            input.seek(nextChunk)
        }
        require(formatFound && dataSize != null) {
            "The backed-up WAV is missing format or audio data."
        }
        val actualDurationMillis = (requireNotNull(dataSize) / 2L) * 1_000L / sampleRate
        val allowedStart = (actualDurationMillis - DURATION_TOLERANCE_MILLIS).coerceAtLeast(0L)
        val allowedEnd = actualDurationMillis + DURATION_TOLERANCE_MILLIS
        require(expectedDurationMillis in allowedStart..allowedEnd) {
            "The backed-up WAV duration does not match its memo metadata."
        }
    }
}

private fun validateWaveformSidecar(
    file: File,
    expectedDurationMillis: Long,
) {
    require(file.length() >= WAVEFORM_HEADER_BYTES) {
        "The backed-up waveform is incomplete."
    }
    DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
        val magic = ByteArray(4).also(input::readFully)
        require(magic.contentEquals(WAVEFORM_MAGIC)) {
            "The backed-up waveform has an invalid header."
        }
        require(input.readInt() == 1) { "The backed-up waveform version is unsupported." }
        val bucketMillis = input.readInt()
        val bucketCount = input.readInt()
        require(bucketMillis in 1..MAX_WAVEFORM_BUCKET_MILLIS) {
            "The backed-up waveform has an invalid bucket duration."
        }
        require(bucketCount in 1..MAX_WAVEFORM_BUCKETS) {
            "The backed-up waveform has an invalid bucket count."
        }
        require(file.length() == WAVEFORM_HEADER_BYTES + bucketCount * 4L) {
            "The backed-up waveform size does not match its header."
        }
        val expectedBuckets = maxOf(
            1L,
            (expectedDurationMillis + bucketMillis - 1L) / bucketMillis,
        )
        require(bucketCount.toLong() in (expectedBuckets - 1L).coerceAtLeast(1L)..expectedBuckets + 1L) {
            "The backed-up waveform duration does not match its memo metadata."
        }
    }
}

private fun RandomAccessFile.readAscii(length: Int): String {
    val bytes = ByteArray(length)
    readFully(bytes)
    return bytes.toString(Charsets.US_ASCII)
}

private fun RandomAccessFile.readUInt16Le(): Int {
    val low = read()
    val high = read()
    require(low >= 0 && high >= 0) { "The backed-up WAV ended unexpectedly." }
    return low or (high shl 8)
}

private fun RandomAccessFile.readUInt32Le(): Long {
    val b0 = read()
    val b1 = read()
    val b2 = read()
    val b3 = read()
    require(b0 >= 0 && b1 >= 0 && b2 >= 0 && b3 >= 0) {
        "The backed-up WAV ended unexpectedly."
    }
    return (
        b0.toLong() or
            (b1.toLong() shl 8) or
            (b2.toLong() shl 16) or
            (b3.toLong() shl 24)
        ) and 0xffff_ffffL
}

private fun verifyFile(
    file: File,
    expectedBytes: Long,
    expectedSha256: String,
    maximumBytes: Long,
    label: String,
) {
    require(expectedBytes in 0..maximumBytes && file.isFile && file.length() == expectedBytes) {
        "The backed-up $label has an invalid size."
    }
    require(sha256(file) == expectedSha256) {
        "The backed-up $label failed its integrity check."
    }
}

private fun sha256(file: File): String =
    file.inputStream().buffered().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(COPY_BUFFER_BYTES)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            if (count > 0) digest.update(buffer, 0, count)
        }
        digest.digest().toHex()
    }

private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).toHex()

private fun ByteArray.toHex(): String = joinToString(separator = "") { "%02x".format(it) }

private fun requireValidUuid(value: String) {
    require(runCatching { UUID.fromString(value) }.getOrNull()?.toString() == value.lowercase()) {
        "A backup contains an invalid memo identifier."
    }
}

private fun requireValidLanguage(value: String) {
    require(value.isNotEmpty() && value.length <= MAX_LANGUAGE_CHARACTERS) {
        "A backup contains an invalid language code."
    }
    require(value.none { it.isISOControl() }) {
        "A backup contains an invalid language code."
    }
}

private fun isAllowedEntryName(name: String): Boolean {
    if (
        name.isEmpty() ||
        name.startsWith('/') ||
        '\\' in name ||
        name.split('/').any { it.isEmpty() || it == "." || it == ".." }
    ) {
        return false
    }
    if (name == MANIFEST_ENTRY) return true
    val match = MEMO_ENTRY_PATTERN.matchEntire(name) ?: return false
    val id = match.groupValues[1]
    return runCatching { requireValidUuid(id) }.isSuccess
}

private fun safeStagingFile(staging: File, name: String): File {
    val root = staging.canonicalFile
    val target = File(root, name).canonicalFile
    require(target.path.startsWith(root.path + File.separator)) {
        "The backup contains an unsafe path."
    }
    return target
}

private fun maximumEntryBytes(name: String): Long = when {
    name == MANIFEST_ENTRY -> MAX_MANIFEST_BYTES
    name.endsWith("/$METADATA_FILE") -> MAX_METADATA_BYTES
    name.endsWith("/$AUDIO_FILE") -> MAX_AUDIO_BYTES
    name.endsWith("/$WAVEFORM_FILE") -> MAX_WAVEFORM_BYTES
    name.endsWith("/$VERSIONS_FILE") ||
    name.endsWith("/$GENERATED_TRANSCRIPT_FILE") ||
        name.endsWith("/$EDITED_TRANSCRIPT_FILE") ||
        name.endsWith("/$EFFECTIVE_TRANSCRIPT_FILE") -> MAX_TRANSCRIPT_BYTES
    else -> 0L
}

private fun isTranscriptEntry(name: String): Boolean =
    name.endsWith("/$VERSIONS_FILE") ||
    name.endsWith("/$GENERATED_TRANSCRIPT_FILE") ||
        name.endsWith("/$EDITED_TRANSCRIPT_FILE") ||
        name.endsWith("/$EFFECTIVE_TRANSCRIPT_FILE")

private const val FORMAT_ID = "com.example.voicememo.backup"
private const val FORMAT_VERSION = 1
private const val MANIFEST_ENTRY = "voice-memo-backup.properties"
private const val MEMOS_DIRECTORY = "memos"
private const val VERSIONS_FILE = "transcript-versions.properties"
private const val METADATA_FILE = "metadata.properties"
private const val AUDIO_FILE = "audio.wav"
private const val WAVEFORM_FILE = "waveform.vwf"
private const val GENERATED_TRANSCRIPT_FILE = "generated-transcript.txt"
private const val EDITED_TRANSCRIPT_FILE = "edited-transcript.txt"
private const val EFFECTIVE_TRANSCRIPT_FILE = "transcript.txt"
private const val PENDING_RESTORE_MARKER = ".voice-memo-restore-pending"
private const val MAX_MEMOS = 10_000
private const val MAX_ENTRIES = 70_001
private const val MAX_TITLE_CHARACTERS = 240
private const val MAX_GENERATED_SUMMARY_CHARACTERS = 220
private const val MAX_METADATA_MODEL_ID_CHARACTERS = 120
private const val MAX_PROJECT_NAME_CHARACTERS = 80
private const val MAX_LANGUAGE_CHARACTERS = 32
private const val MAX_ID_ATTEMPTS = 100
private const val COPY_BUFFER_BYTES = 64 * 1024
private const val MAX_MANIFEST_BYTES = 2L * 1024L * 1024L
private const val MAX_METADATA_BYTES = 1L * 1024L * 1024L
private const val MAX_TRANSCRIPT_BYTES = 16L * 1024L * 1024L
private const val MAX_TOTAL_TRANSCRIPT_BYTES = 32L * 1024L * 1024L
private const val MAX_WAVEFORM_BYTES = 16L * 1024L * 1024L
private const val MAX_AUDIO_BYTES = 5L * 1024L * 1024L * 1024L
private const val MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16L * 1024L * 1024L * 1024L
private const val MIN_FREE_SPACE_RESERVE_BYTES = 512L * 1024L * 1024L
private const val FREE_SPACE_RESERVE_DIVISOR = 10L
private const val MIN_WAV_BYTES = 44L
private const val MIN_AUDIO_SAMPLE_RATE = 8_000L
private const val MAX_AUDIO_SAMPLE_RATE = 384_000L
private const val DURATION_TOLERANCE_MILLIS = 1_000L
private const val WAVEFORM_HEADER_BYTES = 16L
private const val MAX_WAVEFORM_BUCKET_MILLIS = 10_000
private const val MAX_WAVEFORM_BUCKETS = 2_000_000
private val WAVEFORM_MAGIC = byteArrayOf(
    'V'.code.toByte(),
    'W'.code.toByte(),
    'F'.code.toByte(),
    '1'.code.toByte(),
)
private val SHA256_PATTERN = Regex("[0-9a-f]{64}")
private val MEMO_ENTRY_PATTERN = Regex(
    "memos/([0-9A-Fa-f-]{36})/" +
        "(metadata\\.properties|audio\\.wav|waveform\\.vwf|" +
        "transcript-versions\\.properties|generated-transcript\\.txt|edited-transcript\\.txt|transcript\\.txt)",
)
