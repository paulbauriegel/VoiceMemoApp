package com.example.voicememo.backup

import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.SpeakerPerson
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.data.WaveformState
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Properties
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SpeakerBackupTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    private val memoId = "11111111-1111-4111-8111-111111111111"
    private val me = SpeakerPerson("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "Me", true, 10)
    private val audioName = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb.wav"

    private fun samples() = listOf(
        SpeakerVoiceSample(
            id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            personId = me.id,
            modelId = "wespeaker-voxceleb-campplus",
            vector = byteArrayOf(0, 0, -128, 63, 1, 2, 3, 4),
            origin = SpeakerVoiceSample.ORIGIN_RECORDED,
            memoId = null,
            startMillis = null,
            endMillis = null,
            inputChannel = "device-type-15",
            audioFileName = audioName,
            speechMillis = 18_000,
            createdAtMillis = 20,
        ),
        SpeakerVoiceSample(
            id = "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            personId = me.id,
            modelId = "wespeaker-voxceleb-campplus",
            vector = byteArrayOf(9, 8, 7, 6),
            origin = SpeakerVoiceSample.ORIGIN_SEGMENT,
            memoId = memoId,
            startMillis = 1_000,
            endMillis = 4_000,
            inputChannel = null,
            audioFileName = null,
            speechMillis = 3_000,
            createdAtMillis = 30,
        ),
    )

    private class FakeSpeakers(
        var backup: SpeakerBackup,
        val audio: Map<String, File> = emptyMap(),
    ) : SpeakerBackupStore {
        var restored: SpeakerBackup? = null
        var restoredAudio: Map<String, ByteArray> = emptyMap()
        var memoIds: Map<String, String> = emptyMap()

        override suspend fun snapshot() = backup
        override fun sampleAudio(fileName: String) = audio[fileName]
        override suspend fun restore(backup: SpeakerBackup, stagedAudio: Map<String, File>, memoIds: Map<String, String>): Int {
            restored = backup
            restoredAudio = stagedAudio.mapValues { it.value.readBytes() }
            this.memoIds = memoIds
            return backup.samples.size
        }
    }

    private class Memos(val snapshots: List<BackupMemoSnapshot> = emptyList()) : VoiceMemoBackupStore {
        val inserted = ArrayList<RestoredMemo>()
        override suspend fun snapshots() = snapshots
        override suspend fun everyMemoId() = inserted.map { it.memo.id }.toSet()
        override suspend fun insertRestored(memos: List<RestoredMemo>) { inserted += memos }
    }

    private fun wav(): ByteArray {
        val data = ByteArray(3_200)
        return ByteBuffer.allocate(44 + data.size).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + data.size); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1)
            putInt(16_000); putInt(32_000); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(data.size); put(data)
        }.array()
    }

    private fun memoSnapshot(root: File): BackupMemoSnapshot {
        val directory = File(root, memoId).apply { mkdirs() }
        val audio = File(directory, "audio.wav").apply { writeBytes(wav()) }
        return BackupMemoSnapshot(
            memo = MemoRecord(
                id = memoId,
                title = "Memo",
                createdAtMillis = 1,
                updatedAtMillis = 2,
                durationMillis = 100,
                audioPath = audio.absolutePath,
                waveformPath = null,
                waveformState = WaveformState.PENDING,
                languageOverride = null,
            ),
            transcript = null,
        )
    }

    private fun export(root: File, speakers: SpeakerBackupStore?): ByteArray {
        val archive = ByteArrayOutputStream()
        kotlinx.coroutines.runBlocking {
            DefaultVoiceMemoBackupManager(
                store = Memos(listOf(memoSnapshot(root))),
                memoRoot = root,
                speakers = speakers,
                usableSpace = { Long.MAX_VALUE },
            ).exportBackup(archive)
        }
        return archive.toByteArray()
    }

    private fun entries(archive: ByteArray): Map<String, ByteArray> {
        val result = LinkedHashMap<String, ByteArray>()
        ZipInputStream(ByteArrayInputStream(archive)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                result[entry.name] = zip.readBytes()
            }
        }
        return result
    }

    private fun repack(entries: Map<String, ByteArray>): ByteArray {
        val output = ByteArrayOutputStream()
        ZipOutputStream(output).use { zip ->
            entries.forEach { (name, bytes) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(bytes)
                zip.closeEntry()
            }
        }
        return output.toByteArray()
    }

    private fun version(archive: ByteArray): String = Properties().apply {
        load(ByteArrayInputStream(entries(archive).getValue("voice-memo-backup.properties")))
    }.getProperty("version")

    @Test
    fun voiceProfileSurvivesExportAndRestore() = runTest {
        val root = temporaryFolder.newFolder("source")
        val sampleAudio = temporaryFolder.newFile(audioName).apply { writeBytes(wav()) }
        val archive = export(root, FakeSpeakers(SpeakerBackup(listOf(me), samples()), mapOf(audioName to sampleAudio)))

        assertEquals("2", version(archive))
        assertTrue(entries(archive).containsKey("speakers/samples/$audioName"))

        val target = FakeSpeakers(SpeakerBackup(emptyList(), emptyList()))
        val memos = Memos()
        val result = DefaultVoiceMemoBackupManager(
            store = memos,
            memoRoot = temporaryFolder.newFolder("restored"),
            speakers = target,
            usableSpace = { Long.MAX_VALUE },
        ).importBackup(ByteArrayInputStream(archive))

        assertEquals(2, result.restoredVoiceSamples)
        assertEquals(listOf(memoId), result.importedMemoIds)
        val restored = requireNotNull(target.restored)
        assertEquals(listOf("Me"), restored.persons.map { it.name })
        assertTrue(restored.persons.single().isSelf)
        val expected = samples()
        assertEquals(expected.map { it.id }, restored.samples.map { it.id })
        expected.zip(restored.samples).forEach { (before, after) ->
            assertArrayEquals(before.vector, after.vector)
            assertEquals(before.origin, after.origin)
            assertEquals(before.memoId, after.memoId)
            assertEquals(before.startMillis, after.startMillis)
            assertEquals(before.inputChannel, after.inputChannel)
            assertEquals(before.speechMillis, after.speechMillis)
            assertEquals(before.audioFileName, after.audioFileName)
        }
        assertArrayEquals(wav(), target.restoredAudio.getValue(audioName))
        assertEquals(mapOf(memoId to memoId), target.memoIds)
    }

    @Test
    fun backupWithoutVoiceSamplesStaysVersionOne() {
        val root = temporaryFolder.newFolder("plain")
        val withEmptyProfile = export(root, FakeSpeakers(SpeakerBackup(listOf(me), emptyList())))
        assertEquals("1", version(withEmptyProfile))
        assertFalse(entries(withEmptyProfile).keys.any { it.startsWith("speakers/") })
        assertEquals("1", version(export(temporaryFolder.newFolder("plain2"), null)))
    }

    @Test
    fun tamperedVoiceSampleIsRejectedBeforeAnythingIsRestored() = runTest {
        val root = temporaryFolder.newFolder("tamper")
        val sampleAudio = temporaryFolder.newFile(audioName).apply { writeBytes(wav()) }
        val archive = export(root, FakeSpeakers(SpeakerBackup(listOf(me), samples()), mapOf(audioName to sampleAudio)))
        val tampered = entries(archive).toMutableMap()
        tampered["speakers/samples/$audioName"] = wav().also { it[100] = 1 }

        val target = FakeSpeakers(SpeakerBackup(emptyList(), emptyList()))
        val memos = Memos()
        try {
            DefaultVoiceMemoBackupManager(
                store = memos,
                memoRoot = temporaryFolder.newFolder("tamper-restored"),
                speakers = target,
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(repack(tampered)))
            fail("A tampered voice sample was accepted.")
        } catch (expected: VoiceMemoBackupException) {
            assertTrue(expected.message.orEmpty().contains("voice sample"))
        }
        assertNull(target.restored)
        assertTrue(memos.inserted.isEmpty())
    }

    @Test
    fun voiceProfileInAVersionOneArchiveIsRejected() = runTest {
        val root = temporaryFolder.newFolder("downgrade")
        val archive = export(root, FakeSpeakers(SpeakerBackup(listOf(me), samples().drop(1))))
        val downgraded = entries(archive).toMutableMap()
        downgraded["voice-memo-backup.properties"] = downgraded.getValue("voice-memo-backup.properties")
            .toString(Charsets.ISO_8859_1).replace("version=2", "version=1").toByteArray(Charsets.ISO_8859_1)
        try {
            DefaultVoiceMemoBackupManager(
                store = Memos(),
                memoRoot = temporaryFolder.newFolder("downgrade-restored"),
                speakers = FakeSpeakers(SpeakerBackup(emptyList(), emptyList())),
                usableSpace = { Long.MAX_VALUE },
            ).importBackup(ByteArrayInputStream(repack(downgraded)))
            fail("A voice profile in a version 1 archive was accepted.")
        } catch (expected: VoiceMemoBackupException) {
            assertTrue(expected.message.orEmpty().contains("voice profile"))
        }
    }

    @Test
    fun appWithoutSpeakerSupportStillRestoresTheRecordings() = runTest {
        val root = temporaryFolder.newFolder("no-support")
        val archive = export(root, FakeSpeakers(SpeakerBackup(listOf(me), samples().drop(1))))
        val memos = Memos()
        val result = DefaultVoiceMemoBackupManager(
            store = memos,
            memoRoot = temporaryFolder.newFolder("no-support-restored"),
            speakers = null,
            usableSpace = { Long.MAX_VALUE },
        ).importBackup(ByteArrayInputStream(archive))
        assertEquals(1, memos.inserted.size)
        assertEquals(0, result.restoredVoiceSamples)
    }
}
