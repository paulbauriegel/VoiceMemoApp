package com.example.voicememo.data

import androidx.room.Room
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class TranscriptVersionsRepositoryTest {
    @Test
    fun newResultsPreserveEditedAlternativesAndSelectionSurvivesRepositoryRecreation() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val database = Room.inMemoryDatabaseBuilder(context, VoiceMemoDatabase::class.java).build()
        val hooks = object : MemoDeletionHooks {
            override suspend fun cancelTranscriptions(memoIds: Set<String>) = Unit
            override suspend fun deletePrivateFiles(memos: List<MemoRecord>) = Unit
        }
        try {
            val repository = RoomMemoRepository(database, hooks)
            val memo = repository.createMemo(NewMemo(
                title = "Versions", createdAtMillis = 1, durationMillis = 1_000,
                audioPath = "/unused/audio.wav", waveformPath = null, waveformState = WaveformState.PENDING,
            ))
            repository.saveGeneratedTranscript(memo.id, "First result", "en", 10)
            repository.saveEditedTranscript(memo.id, "", 20)
            repository.saveGeneratedTranscript(memo.id, "Second result", "de", 30)
            val second = repository.observeMemo(memo.id).first()!!
            assertEquals("Second result", second.effectiveTranscript)
            assertEquals(2, second.versions.size)
            val firstId = second.versions.single { !it.selected }.id
            repository.selectTranscriptVersion(memo.id, firstId)
            assertEquals("", repository.observeMemo(memo.id).first()!!.effectiveTranscript)
            repository.saveEditedTranscript(memo.id, "Corrected first result", 40)
            repository.selectTranscriptVersion(memo.id, second.versions.single { it.selected }.id)
            assertEquals("Second result", repository.observeMemo(memo.id).first()!!.effectiveTranscript)
            repository.selectTranscriptVersion(memo.id, firstId)
            val reopened = RoomMemoRepository(database, hooks).observeMemo(memo.id).first()!!
            assertEquals("Corrected first result", reopened.effectiveTranscript)
            assertEquals(firstId, reopened.versions.single { it.selected }.id)
            assertEquals("Corrected first result",
                repository.observeLibrary().first().single().effectiveTranscript)
        } finally {
            database.close()
        }
    }
}
