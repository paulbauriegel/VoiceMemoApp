package com.example.voicememo.backup

import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.ProjectIcon
import com.example.voicememo.data.ProjectRecord
import com.example.voicememo.data.TranscriptVersion
import com.example.voicememo.data.TranscriptRecord
import com.example.voicememo.data.WaveformState
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Properties
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class VoiceMemoBackupManagerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun exportThenRestorePreservesAudioMetadataAndTranscriptVersions() = runTest {
        val sourceRoot = temporaryFolder.newFolder("source-memos")
        val originalId = "11111111-1111-4111-8111-111111111111"
        val audioBytes = validPcm16Wav()
        val waveformBytes = validWaveformSidecar()
        val snapshot = createSnapshot(
            root = sourceRoot,
            id = originalId,
            title = "Projekt – 日本語",
            audioBytes = audioBytes,
            waveformBytes = waveformBytes,
            transcript = TranscriptRecord(
                memoId = originalId,
                generatedText = "Grüße 世界\nSecond line",
                editedText = "",
                detectedLanguage = "en",
                generatedAtMillis = 1_700_000_000_200L,
                editedAtMillis = 1_700_000_000_300L,
                generatedRevision = 4,
            ),
            project = ProjectRecord(
                id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                name = "Customer research",
                createdAtMillis = 1_699_000_000_000L,
                updatedAtMillis = 1_700_000_000_100L,
                icon = ProjectIcon.LIGHTBULB,
                isArchived = true,
            ),
        )
        val history = listOf(
            TranscriptVersion("old", originalId, null, "old-model", false,
                requireNotNull(snapshot.transcript).copy(generatedText = "Earlier alternative", editedText = null)),
            TranscriptVersion("chosen", originalId, null, "new-model", true, requireNotNull(snapshot.transcript)),
        )
        val exportStore = FakeBackupStore(initialSnapshots = listOf(snapshot.copy(versions = history)))
        val archive = ByteArrayOutputStream()
        DefaultVoiceMemoBackupManager(
            store = exportStore,
            memoRoot = sourceRoot,
            usableSpace = { Long.MAX_VALUE },
        ).exportBackup(archive)

        val entries = zipEntries(archive.toByteArray())
        assertTrue(entries.keys.contains("voice-memo-backup.properties"))
        assertArrayEquals(
            ByteArray(0),
            entries["memos/$originalId/transcript.txt"],
        )
        assertEquals(
            "Grüße 世界\nSecond line",
            entries["memos/$originalId/generated-transcript.txt"]
                ?.toString(Charsets.UTF_8),
        )

        val restoredRoot = temporaryFolder.newFolder("restored-memos")
        val restoreStore = FakeBackupStore()
        val result = DefaultVoiceMemoBackupManager(
            store = restoreStore,
            memoRoot = restoredRoot,
            usableSpace = { Long.MAX_VALUE },
        ).importBackup(ByteArrayInputStream(archive.toByteArray()))

        assertEquals(listOf(originalId), result.importedMemoIds)
        assertEquals(0, result.remappedMemoCount)
        val restored = restoreStore.inserted.single()
        assertEquals(snapshot.memo.title, restored.memo.title)
        assertEquals(snapshot.memo.createdAtMillis, restored.memo.createdAtMillis)
        assertEquals(snapshot.memo.updatedAtMillis, restored.memo.updatedAtMillis)
        assertEquals(snapshot.memo.durationMillis, restored.memo.durationMillis)
        assertEquals(snapshot.memo.languageOverride, restored.memo.languageOverride)
        assertEquals(snapshot.memo.isStarred, restored.memo.isStarred)
        assertEquals(snapshot.project, restored.project)
        assertEquals(snapshot.project?.id, restored.memo.projectId)
        assertArrayEquals(audioBytes, File(restored.memo.audioPath).readBytes())
        assertArrayEquals(
            waveformBytes,
            File(requireNotNull(restored.memo.waveformPath)).readBytes(),
        )
        assertEquals("Grüße 世界\nSecond line", restored.transcript?.generatedText)
        assertEquals("", restored.transcript?.editedText)
        assertEquals("", restored.transcript?.effectiveText)
        assertEquals(4L, restored.transcript?.generatedRevision)
        assertEquals(history.map { it.transcript }, restored.versions.map { it.transcript })
        assertEquals(listOf(false, true), restored.versions.map { it.selected })
        assertEquals(listOf("old-model", "new-model"), restored.versions.map { it.modelId })
    }

    @Test
    fun restoreRemapsConflictingMemoIdAndKeepsExistingMemoUntouched() = runTest {
        val sourceRoot = temporaryFolder.newFolder("conflict-source")
        val originalId = "22222222-2222-4222-8222-222222222222"
        val replacementId = "33333333-3333-4333-8333-333333333333"
        val snapshot = createSnapshot(sourceRoot, originalId)
        val archive = ByteArrayOutputStream()
        DefaultVoiceMemoBackupManager(
            store = FakeBackupStore(initialSnapshots = listOf(snapshot)),
            memoRoot = sourceRoot,
            usableSpace = { Long.MAX_VALUE },
        ).exportBackup(archive)

        val targetRoot = temporaryFolder.newFolder("conflict-target")
        val existingDirectory = File(targetRoot, originalId).apply { mkdirs() }
        val existingAudio = File(existingDirectory, "audio.wav").apply {
            writeBytes(byteArrayOf(9, 9, 9))
        }
        val targetStore = FakeBackupStore(initialIds = setOf(originalId))
        val result = DefaultVoiceMemoBackupManager(
            store = targetStore,
            memoRoot = targetRoot,
            newId = { replacementId },
            usableSpace = { Long.MAX_VALUE },
        ).importBackup(ByteArrayInputStream(archive.toByteArray()))

        assertEquals(listOf(replacementId), result.importedMemoIds)
        assertEquals(1, result.remappedMemoCount)
        assertArrayEquals(byteArrayOf(9, 9, 9), existingAudio.readBytes())
        assertEquals(replacementId, targetStore.inserted.single().memo.id)
        assertTrue(File(targetRoot, "$replacementId/audio.wav").isFile)
    }

    @Test
    fun restoreRejectsZipSlipBeforeWritingOutsidePrivateStaging() = runTest {
        val root = temporaryFolder.newFolder("zip-slip-target")
        val outside = File(root.parentFile, "escaped.txt")
        val archive = ByteArrayOutputStream().also { bytes ->
            ZipOutputStream(bytes).use { zip ->
                zip.putNextEntry(ZipEntry("../escaped.txt"))
                zip.write("not allowed".toByteArray())
                zip.closeEntry()
            }
        }.toByteArray()
        val store = FakeBackupStore()

        val failure = runCatching {
            DefaultVoiceMemoBackupManager(
                store = store,
                memoRoot = root,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(archive))
        }.exceptionOrNull()

        assertTrue(failure is VoiceMemoBackupException)
        assertFalse(outside.exists())
        assertTrue(store.inserted.isEmpty())
        assertTrue(root.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun restoreRejectsOversizedCompressedMetadataEntry() = runTest {
        val root = temporaryFolder.newFolder("oversized-target")
        val archive = ByteArrayOutputStream().also { bytes ->
            ZipOutputStream(bytes).use { zip ->
                zip.putNextEntry(ZipEntry("voice-memo-backup.properties"))
                zip.write(ByteArray(2 * 1024 * 1024 + 1))
                zip.closeEntry()
            }
        }.toByteArray()
        val store = FakeBackupStore()

        val failure = runCatching {
            DefaultVoiceMemoBackupManager(
                store = store,
                memoRoot = root,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(archive))
        }.exceptionOrNull()

        assertTrue(failure is VoiceMemoBackupException)
        assertTrue(store.inserted.isEmpty())
        assertTrue(root.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun databaseFailureRollsBackEveryPublishedMemoDirectory() = runTest {
        val sourceRoot = temporaryFolder.newFolder("rollback-source")
        val originalId = "44444444-4444-4444-8444-444444444444"
        val archive = ByteArrayOutputStream()
        DefaultVoiceMemoBackupManager(
            store = FakeBackupStore(
                initialSnapshots = listOf(createSnapshot(sourceRoot, originalId)),
            ),
            memoRoot = sourceRoot,
            usableSpace = { Long.MAX_VALUE },
        ).exportBackup(archive)
        val targetRoot = temporaryFolder.newFolder("rollback-target")
        val targetStore = FakeBackupStore(failInsert = true)

        val failure = runCatching {
            DefaultVoiceMemoBackupManager(
                store = targetStore,
                memoRoot = targetRoot,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(archive.toByteArray()))
        }.exceptionOrNull()

        assertNotNull(failure)
        assertTrue(targetStore.inserted.isEmpty())
        assertTrue(targetRoot.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun restoreRejectsArchiveWhoseAudioEntryWasRemoved() = runTest {
        val sourceRoot = temporaryFolder.newFolder("missing-audio-source")
        val originalId = "55555555-5555-4555-8555-555555555555"
        val archive = ByteArrayOutputStream()
        DefaultVoiceMemoBackupManager(
            store = FakeBackupStore(
                initialSnapshots = listOf(createSnapshot(sourceRoot, originalId)),
            ),
            memoRoot = sourceRoot,
            usableSpace = { Long.MAX_VALUE },
        ).exportBackup(archive)
        val withoutAudio = repackWithout(
            archive = archive.toByteArray(),
            omittedEntry = "memos/$originalId/audio.wav",
        )
        val targetRoot = temporaryFolder.newFolder("missing-audio-target")
        val targetStore = FakeBackupStore()

        val failure = runCatching {
            DefaultVoiceMemoBackupManager(
                store = targetStore,
                memoRoot = targetRoot,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(withoutAudio))
        }.exceptionOrNull()

        assertTrue(failure is VoiceMemoBackupException)
        assertTrue(targetStore.inserted.isEmpty())
        assertTrue(targetRoot.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun restoreRejectsNonWavAudioEvenWhenArchiveHashMatches() = runTest {
        val sourceRoot = temporaryFolder.newFolder("invalid-media-source")
        val originalId = "66666666-6666-4666-8666-666666666666"
        val archive = ByteArrayOutputStream()
        DefaultVoiceMemoBackupManager(
            store = FakeBackupStore(
                initialSnapshots = listOf(createSnapshot(sourceRoot, originalId)),
            ),
            memoRoot = sourceRoot,
            usableSpace = { Long.MAX_VALUE },
        ).exportBackup(archive)
        val invalidArchive = replaceAudioWithMatchingMetadata(
            archive = archive.toByteArray(),
            memoId = originalId,
            replacement = "This is not WAV audio".toByteArray(),
        )
        val targetRoot = temporaryFolder.newFolder("invalid-media-target")
        val targetStore = FakeBackupStore()

        val failure = runCatching {
            DefaultVoiceMemoBackupManager(
                store = targetStore,
                memoRoot = targetRoot,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(invalidArchive))
        }.exceptionOrNull()

        assertTrue(failure is VoiceMemoBackupException)
        assertTrue(targetStore.inserted.isEmpty())
        assertTrue(targetRoot.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun nextBackupReconcilesPendingRestoreDirectoriesAfterProcessDeath() = runTest {
        val root = temporaryFolder.newFolder("reconcile-target")
        val orphanId = "77777777-7777-4777-8777-777777777777"
        val committedId = "88888888-8888-4888-8888-888888888888"
        val orphan = File(root, orphanId).apply { mkdirs() }
        File(orphan, ".voice-memo-restore-pending").writeText(orphanId)
        val committed = File(root, committedId).apply { mkdirs() }
        val committedMarker = File(committed, ".voice-memo-restore-pending").apply {
            writeText(committedId)
        }
        val manager = DefaultVoiceMemoBackupManager(
            store = FakeBackupStore(initialIds = setOf(committedId)),
            memoRoot = root,
            usableSpace = { Long.MAX_VALUE },
        )

        manager.exportBackup(ByteArrayOutputStream())

        assertFalse(orphan.exists())
        assertTrue(committed.isDirectory)
        assertFalse(committedMarker.exists())
    }

    private fun createSnapshot(
        root: File,
        id: String,
        title: String = "Project notes",
        audioBytes: ByteArray = validPcm16Wav(),
        waveformBytes: ByteArray? = validWaveformSidecar(),
        transcript: TranscriptRecord? = TranscriptRecord(
            memoId = id,
            generatedText = "Hello from the backup",
            editedText = null,
            detectedLanguage = "en",
            generatedAtMillis = 1_700_000_000_200L,
            editedAtMillis = null,
            generatedRevision = 1,
        ),
        project: ProjectRecord? = null,
    ): BackupMemoSnapshot {
        val directory = File(root, id).apply { mkdirs() }
        val audio = File(directory, "audio.wav").apply { writeBytes(audioBytes) }
        val waveform = waveformBytes?.let {
            File(directory, "waveform.vwf").apply { writeBytes(it) }
        }
        return BackupMemoSnapshot(
            memo = MemoRecord(
                id = id,
                title = title,
                createdAtMillis = 1_700_000_000_000L,
                updatedAtMillis = 1_700_000_000_300L,
                durationMillis = 100L,
                audioPath = audio.absolutePath,
                waveformPath = waveform?.absolutePath,
                waveformState = if (waveform == null) {
                    WaveformState.PENDING
                } else {
                    WaveformState.READY
                },
                languageOverride = "de",
                isStarred = true,
            ),
            transcript = transcript,
            project = project,
        )
    }
}

private class FakeBackupStore(
    private val initialSnapshots: List<BackupMemoSnapshot> = emptyList(),
    initialIds: Set<String> = emptySet(),
    private val failInsert: Boolean = false,
) : VoiceMemoBackupStore {
    private val ids = initialIds.toMutableSet()
    val inserted = mutableListOf<RestoredMemo>()

    override suspend fun snapshots(): List<BackupMemoSnapshot> = initialSnapshots

    override suspend fun everyMemoId(): Set<String> = ids.toSet()

    override suspend fun insertRestored(memos: List<RestoredMemo>) {
        if (failInsert) error("Simulated database failure")
        inserted += memos
        ids += memos.map { it.memo.id }
    }
}

private fun zipEntries(archive: ByteArray): Map<String, ByteArray> {
    val entries = linkedMapOf<String, ByteArray>()
    ZipInputStream(ByteArrayInputStream(archive)).use { zip ->
        while (true) {
            val entry = zip.nextEntry ?: break
            entries[entry.name] = zip.readBytes()
            zip.closeEntry()
        }
    }
    return entries
}

private fun repackWithout(
    archive: ByteArray,
    omittedEntry: String,
): ByteArray {
    val output = ByteArrayOutputStream()
    ZipInputStream(ByteArrayInputStream(archive)).use { input ->
        ZipOutputStream(output).use { zip ->
            while (true) {
                val entry = input.nextEntry ?: break
                if (entry.name != omittedEntry) {
                    zip.putNextEntry(ZipEntry(entry.name))
                    input.copyTo(zip)
                    zip.closeEntry()
                }
                input.closeEntry()
            }
        }
    }
    return output.toByteArray()
}

private fun replaceAudioWithMatchingMetadata(
    archive: ByteArray,
    memoId: String,
    replacement: ByteArray,
): ByteArray {
    val entries = zipEntries(archive).toMutableMap()
    val audioPath = "memos/$memoId/audio.wav"
    val metadataPath = "memos/$memoId/metadata.properties"
    entries[audioPath] = replacement
    val metadata = Properties().apply {
        ByteArrayInputStream(requireNotNull(entries[metadataPath])).use(::load)
    }
    metadata.setProperty("audio.bytes", replacement.size.toString())
    metadata.setProperty(
        "audio.sha256",
        MessageDigest.getInstance("SHA-256")
            .digest(replacement)
            .joinToString("") { "%02x".format(it) },
    )
    entries[metadataPath] = ByteArrayOutputStream().use { output ->
        metadata.store(output, null)
        output.toByteArray()
    }
    return ByteArrayOutputStream().also { output ->
        ZipOutputStream(output).use { zip ->
            entries.forEach { (name, bytes) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(bytes)
                zip.closeEntry()
            }
        }
    }.toByteArray()
}

private fun validPcm16Wav(
    sampleRate: Int = 16_000,
    frameCount: Int = 1_600,
): ByteArray {
    val dataBytes = frameCount * 2
    return ByteBuffer.allocate(44 + dataBytes)
        .order(ByteOrder.LITTLE_ENDIAN)
        .apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(36 + dataBytes)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16)
            putShort(1)
            putShort(1)
            putInt(sampleRate)
            putInt(sampleRate * 2)
            putShort(2)
            putShort(16)
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(dataBytes)
            repeat(frameCount) { putShort(0) }
        }
        .array()
}

private fun validWaveformSidecar(): ByteArray =
    ByteBuffer.allocate(24)
        .apply {
            put(byteArrayOf('V'.code.toByte(), 'W'.code.toByte(), 'F'.code.toByte(), '1'.code.toByte()))
            putInt(1)
            putInt(50)
            putInt(2)
            putShort((-100).toShort())
            putShort(100.toShort())
            putShort((-80).toShort())
            putShort(80.toShort())
        }
        .array()
