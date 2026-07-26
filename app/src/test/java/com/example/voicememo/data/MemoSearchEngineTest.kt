package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MemoSearchEngineTest {
    @Test
    fun partialWordMatchesFromTheStartOrMiddle() {
        assertNotNull(MemoSearchEngine.score("Project planning", "", "proj"))
        assertNotNull(MemoSearchEngine.score("Project planning", "", "ject"))
    }

    @Test
    fun smallTyposAndTransposedLettersMatch() {
        assertNotNull(MemoSearchEngine.score("Project planning", "", "projet"))
        assertNotNull(MemoSearchEngine.score("Weekly recording", "", "recroding"))
    }

    @Test
    fun similarButUnrelatedWordsAreNotFuzzyMatches() {
        assertNull(MemoSearchEngine.score("None", "", "note"))
        assertNull(MemoSearchEngine.score("Regarding", "", "recording"))
        assertNull(MemoSearchEngine.score("Audit notes", "", "audio"))
        assertNull(MemoSearchEngine.score("Votes", "", "notes"))
        assertNull(MemoSearchEngine.score("Conservation", "", "conversation"))
    }

    @Test
    fun accentsAndCommonGermanCharactersAreFolded() {
        assertNotNull(MemoSearchEngine.score("Grüße aus dem Café", "", "grusse cafe"))
    }

    @Test
    fun termsCanMatchInAnyOrderAndAcrossFields() {
        assertNotNull(
            MemoSearchEngine.score(
                title = "Budget meeting",
                transcript = "We need to revisit the launch timeline.",
                query = "timeline budget",
            ),
        )
    }

    @Test
    fun unrelatedTermsDoNotLeakIntoResults() {
        assertNull(MemoSearchEngine.score("Project planning", "Launch timeline", "groceries"))
        assertNull(MemoSearchEngine.score("Project planning", "Launch timeline", "project groceries"))
    }

    @Test
    fun cjkSubstringsRemainSearchable() {
        assertNotNull(MemoSearchEngine.score("会議", "こんにちは世界", "世界"))
    }

    @Test
    fun shortInfixAndPunctuationOnlyQueriesDoNotCreateNoisyResults() {
        assertNull(MemoSearchEngine.score("Project planning", "", "oj"))
        assertTrue(
            MemoSearchEngine.search(
                memos = listOf(
                    summary(
                        id = "memo",
                        title = "Project",
                        transcript = "",
                        createdAt = 1L,
                    ),
                ),
                query = "✨ !!!",
            ).isEmpty(),
        )
    }

    @Test
    fun exactTitleRanksAboveTranscriptAndFuzzyMatches() {
        val exact = summary(id = "exact", title = "Project", transcript = "", createdAt = 1L)
        val transcript = summary(
            id = "transcript",
            title = "Notes",
            transcript = "The project is ready.",
            createdAt = 3L,
        )
        val fuzzy = summary(id = "fuzzy", title = "Projet", transcript = "", createdAt = 2L)

        val ranked = MemoSearchEngine.search(
            memos = listOf(transcript, fuzzy, exact),
            query = "project",
        )

        assertEquals(listOf("exact", "transcript", "fuzzy"), ranked.map(MemoSummary::id))
    }

    @Test
    fun blankQueryPreservesExistingLibraryOrder() {
        val newest = summary(id = "newest", title = "New", transcript = "", createdAt = 2L)
        val oldest = summary(id = "oldest", title = "Old", transcript = "", createdAt = 1L)
        val input = listOf(newest, oldest)

        assertTrue(MemoSearchEngine.search(input, "  ").let { it === input })
    }

    private fun summary(
        id: String,
        title: String,
        transcript: String,
        createdAt: Long,
    ) = MemoSummary(
        id = id,
        title = title,
        createdAtMillis = createdAt,
        updatedAtMillis = createdAt,
        durationMillis = 1_000L,
        audioPath = "/tmp/$id.wav",
        waveformPath = null,
        waveformState = WaveformState.READY,
        languageOverride = null,
        effectiveTranscript = transcript,
        transcriptSource = if (transcript.isBlank()) {
            TranscriptSource.NONE
        } else {
            TranscriptSource.GENERATED
        },
        detectedLanguage = null,
        latestJob = null,
    )
}
