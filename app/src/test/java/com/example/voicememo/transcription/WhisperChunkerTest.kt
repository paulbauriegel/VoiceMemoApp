package com.example.voicememo.transcription

import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperChunkerTest {
    @Test
    fun sixtySeconds_usesTwentyEightSecondWindowsWithTwoSecondOverlap() {
        val windows = WhisperChunker.windows(60 * 16_000)

        assertEquals(3, windows.size)
        assertEquals(listOf(0, 26 * 16_000, 52 * 16_000), windows.map { it.startSample })
        assertEquals(listOf(28 * 16_000, 54 * 16_000, 60 * 16_000), windows.map { it.endSample })
    }

    @Test
    fun tokenMerger_removesLongestBoundaryOverlap() {
        val merged = TokenOverlapMerger.merge(
            left = listOf(10, 11, 12, 13),
            right = listOf(12, 13, 14, 15),
        )

        assertEquals(listOf(10, 11, 12, 13, 14, 15), merged)
    }

    @Test
    fun tokenMerger_preservesRepeatedPhraseAwayFromBoundary() {
        val merged = TokenOverlapMerger.merge(
            left = listOf(1, 2, 1, 2, 3),
            right = listOf(2, 3, 1, 2),
        )

        assertEquals(listOf(1, 2, 1, 2, 3, 1, 2), merged)
    }

    @Test
    fun textMerger_handlesCjkWithoutInventingSpaces() {
        assertEquals("你好世界和平", TextOverlapMerger.merge("你好世界", "世界和平"))
    }

    @Test
    fun truncation_retriesAsTwoHalfWindows() {
        val calls = mutableListOf<Int>()
        val fake = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                calls += audio.size
                return if (audio.size > 14 * 16_000) {
                    WhisperChunkResult(listOf(1), "one", "en", 1, truncated = true)
                } else {
                    val token = if (calls.size == 2) 2 else 3
                    WhisperChunkResult(listOf(token), "$token", "en", 1, truncated = false)
                }
            }

            override fun close() = Unit
        }

        val result = LongFormTranscriber(fake).transcribe(
            speechLikeAudio(28 * 16_000),
            WhisperLanguage.Auto,
        )

        assertEquals(listOf(28 * 16_000, 14 * 16_000, 14 * 16_000), calls)
        assertTrue(result.chunks.single().retriedAsHalfWindows)
        assertFalse(result.chunks.single().result.truncated)
        assertEquals(listOf(2, 3), result.chunks.single().result.tokens)
    }

    @Test
    fun batchedTranscription_loadsOnlyUnfinishedWindowsAndKeepsDetectedLanguage() {
        val windows = WhisperChunker.windows(60 * 16_000)
        val loaded = mutableListOf<Int>()
        val prepared = mutableListOf<Int>()
        val ready = mutableListOf<Int>()
        val languages = mutableListOf<WhisperLanguage>()
        val fake = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                languages += language
                return WhisperChunkResult(
                    tokens = listOf(languages.size + 10),
                    text = "chunk ${languages.size}",
                    detectedLanguage = "de",
                    inferenceMillis = 1,
                    truncated = false,
                )
            }

            override fun close() = Unit
        }
        val completedFirst = LongFormChunk(
            window = windows.first(),
            result = WhisperChunkResult(
                tokens = listOf(10),
                text = "fertig",
                detectedLanguage = "de",
                inferenceMillis = 1,
                truncated = false,
            ),
        )

        val result = LongFormTranscriber(fake).transcribeBatches(
            windows = windows,
            loadAudio = { window ->
                loaded += window.index
                speechLikeAudio(window.sampleCount)
            },
            language = WhisperLanguage.Auto,
            completedChunks = mapOf(0 to completedFirst),
            onChunkPreparing = { window, _ -> prepared += window.index },
            onChunkReady = { window, _ -> ready += window.index },
        )

        assertEquals(listOf(1, 2), loaded)
        assertEquals(listOf(1, 2), prepared)
        assertEquals(listOf(1, 2), ready)
        assertEquals(
            listOf(WhisperLanguage.Forced("de"), WhisperLanguage.Forced("de")),
            languages,
        )
        assertEquals(3, result.chunks.size)
    }

    @Test
    fun batchedTranscription_rejectsIncorrectWindowLength() {
        val window = WhisperChunker.windows(28 * 16_000).single()
        val fake = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ) = WhisperChunkResult(emptyList(), "", "en", 1, truncated = false)

            override fun close() = Unit
        }

        assertThrows(IllegalArgumentException::class.java) {
            LongFormTranscriber(fake).transcribeBatches(
                windows = listOf(window),
                loadAudio = { FloatArray(window.sampleCount - 1) },
                language = WhisperLanguage.Auto,
            )
        }
    }

    @Test
    fun vadSkippedSegment_doesNotInvokeWhisper() {
        var invocations = 0
        val fake = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                invocations++
                return WhisperChunkResult(emptyList(), "", "en", 1, truncated = false)
            }

            override fun close() = Unit
        }

        val transcript = LongFormTranscriber(
            engine = fake,
            voiceActivityDetector = VoiceActivityDetector { _, _ -> false },
        ).transcribe(
            FloatArray(28 * 16_000),
            WhisperLanguage.Forced("en"),
        )

        assertEquals(0, invocations)
        assertTrue(transcript.chunks.single().skippedByVad)
        assertEquals("", transcript.text)
    }

    @Test
    fun disabledVad_sendsSilentSegmentToWhisper() {
        var invocations = 0
        val fake = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                invocations++
                return WhisperChunkResult(
                    tokens = listOf(42),
                    text = "decoded silence",
                    detectedLanguage = "en",
                    inferenceMillis = 1,
                    truncated = false,
                )
            }

            override fun close() = Unit
        }

        val transcript = LongFormTranscriber(fake).transcribe(
            FloatArray(28 * 16_000),
            WhisperLanguage.Forced("en"),
        )

        assertEquals(1, invocations)
        assertFalse(transcript.chunks.single().skippedByVad)
        assertEquals("decoded silence", transcript.text)
    }

    @Test
    fun twoEngines_transcribeTwoSegmentsConcurrentlyAndKeepResultOrder() {
        val windows = WhisperChunker.windows(60 * 16_000)
        val enteredFirstBatch = CountDownLatch(2)
        val activeInference = AtomicInteger()
        val maximumInference = AtomicInteger()
        val activeAudioLoads = AtomicInteger()
        val maximumAudioLoads = AtomicInteger()
        val completed = Collections.synchronizedList(mutableListOf<Int>())

        fun engine() = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                val active = activeInference.incrementAndGet()
                maximumInference.accumulateAndGet(active) { left, right -> maxOf(left, right) }
                enteredFirstBatch.countDown()
                check(enteredFirstBatch.await(2, TimeUnit.SECONDS)) {
                    "Two Whisper segments did not overlap."
                }
                try {
                    val token = audio.first().roundToInt()
                    return WhisperChunkResult(
                        tokens = listOf(token),
                        text = token.toString(),
                        detectedLanguage = "en",
                        inferenceMillis = 1,
                        truncated = false,
                    )
                } finally {
                    activeInference.decrementAndGet()
                }
            }

            override fun close() = Unit
        }

        val result = LongFormTranscriber(listOf(engine(), engine())).transcribeBatches(
            windows = windows,
            loadAudio = { window ->
                val active = activeAudioLoads.incrementAndGet()
                maximumAudioLoads.accumulateAndGet(active) { left, right -> maxOf(left, right) }
                try {
                    Thread.sleep(10)
                    speechLikeAudio(window.sampleCount).also {
                        it[0] = (window.index + 10).toFloat()
                    }
                } finally {
                    activeAudioLoads.decrementAndGet()
                }
            },
            language = WhisperLanguage.Forced("en"),
            onChunkCompleted = { chunk -> completed += chunk.window.index },
        )

        assertEquals(2, maximumInference.get())
        assertEquals(1, maximumAudioLoads.get())
        assertEquals(listOf(0, 1, 2), completed)
        assertEquals(listOf(10, 11, 12), result.tokens)
        assertEquals(listOf(0, 1, 2), result.chunks.map { it.window.index })
    }

    @Test
    fun fastEngine_startsNextSegmentWhileOtherEngineIsStillBusy() {
        val windows = List(3) { index ->
            AudioChunkWindow(
                index = index,
                startSample = index * 16_000,
                endSample = (index + 1) * 16_000,
            )
        }
        val slowSegmentStarted = CountDownLatch(1)
        val releaseSlowSegment = CountDownLatch(1)
        val laterSegmentStarted = CountDownLatch(1)
        val completed = Collections.synchronizedList(mutableListOf<Int>())

        fun engine() = object : WhisperEngine {
            override fun transcribeChunk(
                audio: FloatArray,
                language: WhisperLanguage,
                shouldCancel: () -> Boolean,
                onTokenProgress: (generated: Int, limit: Int) -> Unit,
            ): WhisperChunkResult {
                val segment = audio.first().roundToInt()
                when (segment) {
                    0 -> {
                        slowSegmentStarted.countDown()
                        check(releaseSlowSegment.await(2, TimeUnit.SECONDS)) {
                            "Timed out waiting to release the slow segment."
                        }
                    }

                    2 -> laterSegmentStarted.countDown()
                }
                return WhisperChunkResult(
                    tokens = listOf(segment),
                    text = segment.toString(),
                    detectedLanguage = "en",
                    inferenceMillis = 1,
                    truncated = false,
                )
            }

            override fun close() = Unit
        }

        val caller = Executors.newSingleThreadExecutor()
        try {
            val transcript = caller.submit<LongFormTranscript> {
                LongFormTranscriber(listOf(engine(), engine())).transcribeBatches(
                    windows = windows,
                    loadAudio = { window ->
                        speechLikeAudio(window.sampleCount).also {
                            it[0] = window.index.toFloat()
                        }
                    },
                    language = WhisperLanguage.Forced("en"),
                    onChunkCompleted = { chunk -> completed += chunk.window.index },
                )
            }

            assertTrue(slowSegmentStarted.await(2, TimeUnit.SECONDS))
            assertTrue(
                "A free engine did not claim segment 3 while segment 1 was still running.",
                laterSegmentStarted.await(2, TimeUnit.SECONDS),
            )
            assertEquals(1L, releaseSlowSegment.count)
            releaseSlowSegment.countDown()

            val result = transcript.get(2, TimeUnit.SECONDS)
            assertEquals(listOf(0, 1, 2), completed)
            assertEquals(listOf(0, 1, 2), result.tokens)
        } finally {
            releaseSlowSegment.countDown()
            caller.shutdownNow()
        }
    }

    private fun speechLikeAudio(sampleCount: Int): FloatArray =
        FloatArray(sampleCount) { index ->
            (sin(2.0 * PI * 180.0 * index / 16_000) * 0.1).toFloat()
        }
}
