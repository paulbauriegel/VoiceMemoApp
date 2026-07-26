package com.example.voicememo.importing

import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.NewMemo
import com.example.voicememo.recording.WaveformSidecar
import com.example.voicememo.transcription.WavAudioReader
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class AudioImportTransactionTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun commitsCanonicalWavWaveformAndMemoTogether() = runTest {
        val root = temporaryFolder.newFolder("memos")
        var inserted: NewMemo? = null
        val transaction = AudioImportTransaction(
            memoRoot = root,
            storageIdFactory = { "storage-id" },
            createMemo = { newMemo ->
                inserted = newMemo
                newMemo.toRecord("memo-id")
            },
        )
        val samples = ShortArray(16_000) { index ->
            if (index % 2 == 0) 12_000 else -8_000
        }

        val memo = transaction.import(
            title = "Meeting.v2",
            createdAtMillis = 123L,
        ) { sink ->
            sink.write(16_000, samples, samples.size)
        }

        assertEquals("memo-id", memo.id)
        assertEquals("Meeting.v2", memo.title)
        assertEquals(1_000L, memo.durationMillis)
        assertEquals("storage-id", File(memo.audioPath).parentFile?.name)
        assertTrue(File(memo.audioPath).isFile)
        assertTrue(File(requireNotNull(memo.waveformPath)).isFile)
        assertEquals(16_000, WavAudioReader.read(File(memo.audioPath)).samples.size)
        assertEquals(20, WaveformSidecar.read(File(requireNotNull(memo.waveformPath))).size)
        assertNotNull(inserted)
        assertFalse(root.listFiles().orEmpty().any { it.name.startsWith(".importing-") })
    }

    @Test
    fun repositoryFailureRollsBackPublishedFiles() = runTest {
        val root = temporaryFolder.newFolder("memos")
        val transaction = AudioImportTransaction(
            memoRoot = root,
            storageIdFactory = { "will-rollback" },
            createMemo = { error("Room insert failed") },
        )

        val failure = runCatching {
            transaction.import("Memo", 1L) { sink ->
                val samples = ShortArray(32) { it.toShort() }
                sink.write(16_000, samples, samples.size)
            }
        }.exceptionOrNull()

        assertEquals("Room insert failed", failure?.message)
        assertTrue(root.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun cancellationRemovesIncompleteStage() = runTest {
        val root = temporaryFolder.newFolder("memos")
        var createCalled = false
        val transaction = AudioImportTransaction(
            memoRoot = root,
            storageIdFactory = { "cancelled" },
            createMemo = {
                createCalled = true
                it.toRecord("unexpected")
            },
        )

        val failure = runCatching {
            transaction.import("Memo", 1L) { sink ->
                sink.write(16_000, shortArrayOf(1, 2, 3), 3)
                throw CancellationException("Picker closed")
            }
        }.exceptionOrNull()

        assertTrue(failure is CancellationException)
        assertFalse(createCalled)
        assertTrue(root.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun emptyDecodeCreatesNoMemoOrFiles() = runTest {
        val root = temporaryFolder.newFolder("memos")
        var createCalled = false
        val transaction = AudioImportTransaction(
            memoRoot = root,
            storageIdFactory = { "empty" },
            createMemo = {
                createCalled = true
                it.toRecord("unexpected")
            },
        )

        val failure = runCatching {
            transaction.import("Memo", 1L) { }
        }.exceptionOrNull()

        assertEquals("The selected file contains no decodable audio.", failure?.message)
        assertFalse(createCalled)
        assertTrue(root.listFiles().orEmpty().isEmpty())
    }

    private fun NewMemo.toRecord(id: String) = MemoRecord(
        id = id,
        title = title,
        createdAtMillis = createdAtMillis,
        updatedAtMillis = createdAtMillis,
        durationMillis = durationMillis,
        audioPath = audioPath,
        waveformPath = waveformPath,
        waveformState = waveformState,
        languageOverride = null,
    )
}
