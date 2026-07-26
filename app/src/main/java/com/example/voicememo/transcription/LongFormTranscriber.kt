package com.example.voicememo.transcription

import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

data class AudioChunkWindow(
    val index: Int,
    val startSample: Int,
    val endSample: Int,
) {
    val sampleCount: Int get() = endSample - startSample
}

object WhisperChunker {
    const val PAYLOAD_SECONDS = 28
    const val OVERLAP_SECONDS = 2
    const val PAYLOAD_SAMPLES = PAYLOAD_SECONDS * WhisperFeatureExtractor.SAMPLE_RATE
    const val OVERLAP_SAMPLES = OVERLAP_SECONDS * WhisperFeatureExtractor.SAMPLE_RATE
    const val STRIDE_SAMPLES = PAYLOAD_SAMPLES - OVERLAP_SAMPLES

    fun windows(sampleCount: Int): List<AudioChunkWindow> {
        require(sampleCount >= 0)
        if (sampleCount == 0) return emptyList()
        val result = mutableListOf<AudioChunkWindow>()
        var start = 0
        var index = 0
        while (start < sampleCount) {
            val end = minOf(start + PAYLOAD_SAMPLES, sampleCount)
            result += AudioChunkWindow(index++, start, end)
            if (end == sampleCount) break
            start += STRIDE_SAMPLES
        }
        return result
    }
}

data class LongFormChunk(
    val window: AudioChunkWindow,
    val result: WhisperChunkResult,
    val retriedAsHalfWindows: Boolean = false,
    val skippedByVad: Boolean = false,
)

data class LongFormProgress(
    val chunkIndex: Int,
    val chunkCount: Int,
    val generatedTokens: Int,
    val tokenLimit: Int,
)

data class LongFormTranscript(
    val text: String,
    val tokens: List<Int>,
    val detectedLanguage: String,
    val chunks: List<LongFormChunk>,
    val warnings: List<String>,
)

/**
 * Stateful long-recording orchestration around the fixed 30-second model.
 *
 * Persisted [completedChunks] can be supplied after process death; those
 * windows are not inferred again.
 */
class LongFormTranscriber(
    private val engines: List<WhisperEngine>,
    private val voiceActivityDetector: VoiceActivityDetector? = null,
) {
    constructor(
        engine: WhisperEngine,
        voiceActivityDetector: VoiceActivityDetector? = null,
    ) : this(
        engines = listOf(engine),
        voiceActivityDetector = voiceActivityDetector,
    )

    init {
        require(engines.isNotEmpty()) { "At least one Whisper engine is required." }
    }

    fun transcribe(
        samples16Khz: FloatArray,
        language: WhisperLanguage,
        completedChunks: Map<Int, LongFormChunk> = emptyMap(),
        shouldCancel: () -> Boolean = { false },
        onProgress: (LongFormProgress) -> Unit = {},
        onChunkCompleted: (LongFormChunk) -> Unit = {},
    ): LongFormTranscript = transcribeBatches(
        windows = WhisperChunker.windows(samples16Khz.size),
        loadAudio = { window ->
            samples16Khz.copyOfRange(window.startSample, window.endSample)
        },
        language = language,
        completedChunks = completedChunks,
        shouldCancel = shouldCancel,
        onProgress = onProgress,
        onChunkCompleted = onChunkCompleted,
    )

    /**
     * Transcribes long recordings without requiring the entire recording in
     * memory. [loadAudio] is called only for the current unfinished window, so
     * callers can read and resample directly from a seekable audio source.
     */
    fun transcribeBatches(
        windows: List<AudioChunkWindow>,
        loadAudio: (AudioChunkWindow) -> FloatArray,
        language: WhisperLanguage,
        completedChunks: Map<Int, LongFormChunk> = emptyMap(),
        shouldCancel: () -> Boolean = { false },
        onChunkPreparing: (AudioChunkWindow, Int) -> Unit = { _, _ -> },
        onChunkReady: (AudioChunkWindow, Int) -> Unit = { _, _ -> },
        onProgress: (LongFormProgress) -> Unit = {},
        onChunkCompleted: (LongFormChunk) -> Unit = {},
    ): LongFormTranscript {
        if (windows.isEmpty()) {
            return LongFormTranscript("", emptyList(), "und", emptyList(), emptyList())
        }
        windows.forEachIndexed { position, window ->
            require(window.index == position) {
                "Whisper windows must have contiguous zero-based indices."
            }
            require(window.startSample >= 0 && window.endSample > window.startSample) {
                "Whisper window ${window.index} has an invalid sample range."
            }
        }

        val chunks = arrayOfNulls<LongFormChunk>(windows.size)
        var forcedLanguage = (language as? WhisperLanguage.Forced)?.code
        completedChunks.toSortedMap().forEach { (index, completed) ->
            require(index in windows.indices) {
                "Completed Whisper segment $index is outside the requested windows."
            }
            // Room stores window boundaries in milliseconds, which can round a
            // final partial sample. The durable chunk index is authoritative.
            chunks[index] = completed.copy(window = windows[index])
            if (forcedLanguage == null && completed.result.detectedLanguage != "und") {
                forcedLanguage = completed.result.detectedLanguage
            }
        }

        val unfinished = windows.filter { chunks[it.index] == null }
        var nextUnfinished = 0

        // Automatic language detection must happen before parallel work so every
        // later segment receives the same forced language. Silent leading
        // segments are skipped until the first speech segment establishes it.
        while (forcedLanguage == null && nextUnfinished < unfinished.size) {
            val window = unfinished[nextUnfinished++]
            val chunk = transcribeWindow(
                engine = engines.first(),
                window = window,
                windowCount = windows.size,
                loadAudio = loadAudio,
                language = WhisperLanguage.Auto,
                shouldCancel = shouldCancel,
                onChunkPreparing = onChunkPreparing,
                onChunkReady = onChunkReady,
                onProgress = onProgress,
            )
            chunks[window.index] = chunk
            onChunkCompleted(chunk)
            if (chunk.result.detectedLanguage != "und") {
                forcedLanguage = chunk.result.detectedLanguage
            }
        }

        val remaining = unfinished.drop(nextUnfinished)
        if (remaining.isNotEmpty()) {
            val languageForRemaining =
                forcedLanguage?.let(WhisperLanguage::Forced) ?: WhisperLanguage.Auto
            if (engines.size == 1 || remaining.size == 1) {
                for (window in remaining) {
                    val chunk = transcribeWindow(
                        engine = engines.first(),
                        window = window,
                        windowCount = windows.size,
                        loadAudio = loadAudio,
                        language = languageForRemaining,
                        shouldCancel = shouldCancel,
                        onChunkPreparing = onChunkPreparing,
                        onChunkReady = onChunkReady,
                        onProgress = onProgress,
                    )
                    chunks[window.index] = chunk
                    onChunkCompleted(chunk)
                }
            } else {
                transcribeParallelQueue(
                    windows = remaining,
                    windowCount = windows.size,
                    loadAudio = loadAudio,
                    language = languageForRemaining,
                    shouldCancel = shouldCancel,
                    onChunkPreparing = onChunkPreparing,
                    onChunkReady = onChunkReady,
                    onProgress = onProgress,
                    onResult = { chunk ->
                        chunks[chunk.window.index] = chunk
                        onChunkCompleted(chunk)
                    },
                )
            }
        }

        val orderedChunks = chunks.mapIndexed { index, chunk ->
            requireNotNull(chunk) { "Whisper segment $index did not produce a result." }
        }
        var mergedTokens = emptyList<Int>()
        var mergedText = ""
        for (chunk in orderedChunks) {
            mergedTokens = TokenOverlapMerger.merge(mergedTokens, chunk.result.tokens)
            mergedText = TextOverlapMerger.merge(mergedText, chunk.result.text)
        }
        val containsTimestampTokens = orderedChunks.any { chunk ->
            WhisperTimestampTokens.containsTimestamp(
                chunk.result.tokens,
                WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID,
            )
        }
        val tokenizerMergedText = if (containsTimestampTokens) {
            null
        } else {
            engines.first().decodeTokens(mergedTokens)
        }
        return LongFormTranscript(
            text = tokenizerMergedText?.trim() ?: mergedText.trim(),
            tokens = mergedTokens,
            detectedLanguage = forcedLanguage ?: "und",
            chunks = orderedChunks,
            warnings = orderedChunks.flatMap { it.result.warnings }.distinct(),
        )
    }

    private fun transcribeParallelQueue(
        windows: List<AudioChunkWindow>,
        windowCount: Int,
        loadAudio: (AudioChunkWindow) -> FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onChunkPreparing: (AudioChunkWindow, Int) -> Unit,
        onChunkReady: (AudioChunkWindow, Int) -> Unit,
        onProgress: (LongFormProgress) -> Unit,
        onResult: (LongFormChunk) -> Unit,
    ) {
        val callbackLock = Any()
        val audioLoadLock = Any()
        val nextWindowPosition = AtomicInteger()
        val failureSignalled = AtomicBoolean()
        val completed = arrayOfNulls<LongFormChunk>(windows.size)
        val failures = arrayOfNulls<Throwable>(windows.size)
        val completionSignals = Array(windows.size) { CountDownLatch(1) }
        val executor = Executors.newFixedThreadPool(engines.size) { runnable ->
            Thread(runnable, "Whisper-segment").apply { isDaemon = true }
        }
        try {
            // Each engine owns one worker loop. As soon as an engine completes
            // a segment it claims the next unfinished one, without waiting for
            // another engine or for durable result delivery.
            val workers = engines.map { engine ->
                executor.submit(
                    Callable {
                        while (!failureSignalled.get()) {
                            val position = nextWindowPosition.getAndIncrement()
                            if (position >= windows.size) break
                            val window = windows[position]
                            try {
                                completed[position] =
                                    transcribeWindow(
                                        engine = engine,
                                        window = window,
                                        windowCount = windowCount,
                                        loadAudio = { requestedWindow ->
                                            // StreamingWavChunkSource deliberately has
                                            // one seek cursor. Keep preparation bounded
                                            // and serialized while the expensive model
                                            // inference runs concurrently.
                                            synchronized(audioLoadLock) {
                                                loadAudio(requestedWindow)
                                            }
                                        },
                                        language = language,
                                        shouldCancel = shouldCancel,
                                        onChunkPreparing = { preparing, count ->
                                            synchronized(callbackLock) {
                                                onChunkPreparing(preparing, count)
                                            }
                                        },
                                        onChunkReady = { ready, count ->
                                            synchronized(callbackLock) {
                                                onChunkReady(ready, count)
                                            }
                                        },
                                        onProgress = { progress ->
                                            synchronized(callbackLock) {
                                                onProgress(progress)
                                            }
                                        },
                                    )
                            } catch (failure: Throwable) {
                                failures[position] = failure
                                failureSignalled.set(true)
                                throw failure
                            } finally {
                                completionSignals[position].countDown()
                            }
                        }
                    },
                )
            }

            // Workers publish into position-addressed slots. Waiting and
            // delivering here preserves durable segment order while leaving
            // every engine free to continue inference immediately.
            completionSignals.indices.forEach { position ->
                completionSignals[position].await()
                failures[position]?.let(::throwParallelFailure)
                val chunk = requireNotNull(completed[position]) {
                    "Parallel Whisper segment ${windows[position].index} did not produce a result."
                }
                synchronized(callbackLock) {
                    onResult(chunk)
                }
            }
            workers.forEach { worker -> worker.resultOrThrow() }
        } finally {
            executor.shutdownNow()
        }
    }

    private fun throwParallelFailure(failure: Throwable): Nothing = when (failure) {
        is Error -> throw failure
        is RuntimeException -> throw failure
        else -> throw IllegalStateException("Parallel Whisper segment failed.", failure)
    }

    private fun transcribeWindow(
        engine: WhisperEngine,
        window: AudioChunkWindow,
        windowCount: Int,
        loadAudio: (AudioChunkWindow) -> FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onChunkPreparing: (AudioChunkWindow, Int) -> Unit,
        onChunkReady: (AudioChunkWindow, Int) -> Unit,
        onProgress: (LongFormProgress) -> Unit,
    ): LongFormChunk {
        if (shouldCancel()) throw WhisperCancelledException()
        onChunkPreparing(window, windowCount)
        val audio = loadAudio(window)
        require(audio.size == window.sampleCount) {
            "Whisper window ${window.index} expected ${window.sampleCount} samples, " +
                "but the audio source returned ${audio.size}."
        }
        if (shouldCancel()) throw WhisperCancelledException()
        if (voiceActivityDetector?.hasSpeech(audio, shouldCancel) == false) {
            return LongFormChunk(
                window = window,
                result = WhisperChunkResult(
                    tokens = emptyList(),
                    text = "",
                    detectedLanguage = (language as? WhisperLanguage.Forced)?.code ?: "und",
                    inferenceMillis = 0,
                    truncated = false,
                ),
                skippedByVad = true,
            )
        }

        onChunkReady(window, windowCount)
        val reportProgress = { generated: Int, limit: Int ->
            onProgress(
                LongFormProgress(
                    chunkIndex = window.index,
                    chunkCount = windowCount,
                    generatedTokens = generated,
                    tokenLimit = limit,
                ),
            )
        }
        val initial = engine.transcribeChunk(
            audio = audio,
            language = language,
            shouldCancel = shouldCancel,
            onTokenProgress = reportProgress,
        )
        return if (initial.truncated) {
            retryTruncated(
                engine = engine,
                window = window,
                audio = audio,
                language = initial.detectedLanguage.takeUnless { it == "und" }
                    ?.let(WhisperLanguage::Forced)
                    ?: language,
                shouldCancel = shouldCancel,
                onTokenProgress = reportProgress,
            )
        } else {
            LongFormChunk(window, initial)
        }
    }

    private fun <T> Future<T>.resultOrThrow(): T = try {
        get()
    } catch (failure: ExecutionException) {
        when (val cause = failure.cause ?: failure) {
            is Error -> throw cause
            is RuntimeException -> throw cause
            else -> throw IllegalStateException("Parallel Whisper segment failed.", cause)
        }
    }

    private fun retryTruncated(
        engine: WhisperEngine,
        window: AudioChunkWindow,
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (Int, Int) -> Unit,
    ): LongFormChunk {
        val midpoint = minOf(
            14 * WhisperFeatureExtractor.SAMPLE_RATE,
            audio.size,
        )
        val first = engine.transcribeChunk(
            audio.copyOfRange(0, midpoint),
            language,
            shouldCancel,
            onTokenProgress,
        )
        val second = if (midpoint < audio.size) {
            engine.transcribeChunk(
                audio.copyOfRange(midpoint, audio.size),
                WhisperLanguage.Forced(first.detectedLanguage),
                shouldCancel,
                onTokenProgress,
            )
        } else {
            WhisperChunkResult(
                emptyList(),
                "",
                first.detectedLanguage,
                0L,
                truncated = false,
            )
        }
        val remainsTruncated = first.truncated || second.truncated
        val secondTokens = if (
            WhisperTimestampTokens.containsTimestamp(
                second.tokens,
                WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID,
            )
        ) {
            WhisperTimestampTokens.shift(
                tokens = second.tokens,
                timestampBeginTokenId =
                    WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID,
                offsetMillis = midpoint * 1_000L / WhisperFeatureExtractor.SAMPLE_RATE,
            )
        } else {
            second.tokens
        }
        val combinedTokens = buildList(first.tokens.size + secondTokens.size) {
            addAll(first.tokens)
            addAll(secondTokens)
        }
        val combinedText = if (
            WhisperTimestampTokens.containsTimestamp(
                combinedTokens,
                WhisperTimestampTokens.DEFAULT_TIMESTAMP_BEGIN_TOKEN_ID,
            )
        ) {
            TextConcatenator.concatenate(first.text, second.text)
        } else {
            engine.decodeTokens(combinedTokens)?.trim()
                ?: TextConcatenator.concatenate(first.text, second.text)
        }
        val warnings = buildList {
            add("A 28-second chunk reached the token limit and was retried as two 14-second windows.")
            addAll(first.warnings)
            addAll(second.warnings)
            if (remainsTruncated) {
                add("Transcription may be incomplete because a 14-second retry also reached 128 tokens.")
            }
        }.distinct()
        return LongFormChunk(
            window = window,
            result = WhisperChunkResult(
                // The two fallback windows are adjacent, not overlapping.
                tokens = combinedTokens,
                text = combinedText,
                detectedLanguage = first.detectedLanguage,
                inferenceMillis = first.inferenceMillis + second.inferenceMillis,
                truncated = remainsTruncated,
                warnings = warnings,
                timedSegments = buildList {
                    addAll(first.timedSegments)
                    val offsetMillis =
                        midpoint * 1_000L / WhisperFeatureExtractor.SAMPLE_RATE
                    addAll(
                        second.timedSegments.map { segment ->
                            segment.copy(
                                startMillis = segment.startMillis + offsetMillis,
                                endMillis = segment.endMillis + offsetMillis,
                            )
                        },
                    )
                },
            ),
            retriedAsHalfWindows = true,
        )
    }

}

object TokenOverlapMerger {
    private const val MAX_OVERLAP_TOKENS = 32

    fun merge(left: List<Int>, right: List<Int>): List<Int> {
        if (left.isEmpty()) return right.toList()
        if (right.isEmpty()) return left.toList()
        val maximum = minOf(left.size, right.size, MAX_OVERLAP_TOKENS)
        var overlap = maximum
        while (overlap > 0) {
            var matches = true
            for (index in 0 until overlap) {
                if (left[left.size - overlap + index] != right[index]) {
                    matches = false
                    break
                }
            }
            if (matches) break
            overlap--
        }
        return buildList(left.size + right.size - overlap) {
            addAll(left)
            addAll(right.subList(overlap, right.size))
        }
    }
}

private object TextConcatenator {
    fun concatenate(left: String, right: String): String {
        val lhs = left.trimEnd()
        val rhs = right.trimStart()
        if (lhs.isEmpty()) return rhs
        if (rhs.isEmpty()) return lhs
        val separator = if (
            lhs.last().isWhitespace() ||
            rhs.first().isWhitespace() ||
            (isCjkCharacter(lhs.last()) && isCjkCharacter(rhs.first()))
        ) {
            ""
        } else {
            " "
        }
        return lhs + separator + rhs
    }

}

internal object TextOverlapMerger {
    private const val MAX_OVERLAP_CHARACTERS = 160

    fun merge(left: String, right: String): String {
        if (left.isBlank()) return right.trim()
        if (right.isBlank()) return left.trim()
        val lhs = left.trimEnd()
        val rhs = right.trimStart()
        val maximum = minOf(lhs.length, rhs.length, MAX_OVERLAP_CHARACTERS)
        var overlap = maximum
        while (overlap > 0 && !lhs.regionMatches(lhs.length - overlap, rhs, 0, overlap)) {
            overlap--
        }
        val tail = rhs.substring(overlap)
        if (tail.isEmpty()) return lhs
        val separator = if (needsSpace(lhs.last(), tail.first())) " " else ""
        return lhs + separator + tail
    }

    private fun needsSpace(left: Char, right: Char): Boolean {
        if (left.isWhitespace() || right.isWhitespace()) return false
        if (left in "([{“‘\"'" || right in ".,!?;:)]}”’\"'") return false
        if (isCjkCharacter(left) && isCjkCharacter(right)) return false
        return true
    }
}

/**
 * API-23-safe BMP ranges used only to decide whether a separator space is
 * appropriate. Whisper text remains untouched.
 */
private fun isCjkCharacter(character: Char): Boolean = character.code in 0x3400..0x4DBF ||
    character.code in 0x4E00..0x9FFF ||
    character.code in 0xF900..0xFAFF ||
    character.code in 0x3040..0x30FF ||
    character.code in 0xAC00..0xD7AF
