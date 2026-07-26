package com.example.voicememo.transcription

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Offline Silero VAD v6.2.1 inference.
 *
 * The recurrent state and 64-sample context are reset for every Whisper
 * window. Calls are synchronized because parallel Whisper workers share this
 * small CPU session while each window needs independent recurrent state.
 */
internal class SileroVoiceActivityDetector private constructor(
    private val session: OrtSession,
    private val inputTensor: OnnxTensor,
    private val stateTensor: OnnxTensor,
    private val sampleRateTensor: OnnxTensor,
    private val inputBuffer: FloatBuffer,
    private val stateBuffer: FloatBuffer,
) : VoiceActivityDetector, AutoCloseable {
    private val recurrentState = FloatArray(STATE_FLOAT_COUNT)
    private val context = FloatArray(CONTEXT_SAMPLES)
    private val frame = FloatArray(FRAME_SAMPLES)

    @Synchronized
    override fun hasSpeech(
        audio16Khz: FloatArray,
        shouldCancel: () -> Boolean,
    ): Boolean {
        if (audio16Khz.isEmpty()) return false
        recurrentState.fill(0f)
        context.fill(0f)
        val speechGate = SileroSpeechGate()

        return try {
            var offset = 0
            while (offset < audio16Khz.size) {
                if (shouldCancel()) throw WhisperCancelledException()
                frame.fill(0f)
                val sampleCount = minOf(FRAME_SAMPLES, audio16Khz.size - offset)
                audio16Khz.copyInto(
                    destination = frame,
                    destinationOffset = 0,
                    startIndex = offset,
                    endIndex = offset + sampleCount,
                )

                inputBuffer.clear()
                inputBuffer.put(context)
                inputBuffer.put(frame)
                inputBuffer.rewind()
                stateBuffer.clear()
                stateBuffer.put(recurrentState)
                stateBuffer.rewind()

                session.run(
                    mapOf(
                        INPUT_NAME to inputTensor,
                        STATE_NAME to stateTensor,
                        SAMPLE_RATE_NAME to sampleRateTensor,
                    ),
                ).use { result ->
                    val probabilityTensor = result[0] as OnnxTensor
                    val nextStateTensor = result[1] as OnnxTensor
                    val probability = probabilityTensor.floatBuffer.get(0)
                    val nextState = nextStateTensor.floatBuffer
                    nextState.get(recurrentState)
                    if (speechGate.observe(probability)) return true
                }

                frame.copyInto(
                    destination = context,
                    startIndex = FRAME_SAMPLES - CONTEXT_SAMPLES,
                    endIndex = FRAME_SAMPLES,
                    destinationOffset = 0,
                )
                offset += sampleCount
            }
            false
        } catch (cancelled: WhisperCancelledException) {
            throw cancelled
        } catch (failure: Exception) {
            // VAD is an optimization. A runtime problem must never suppress
            // possibly valid speech, so keep the audio and let Whisper decide.
            Log.w(TAG, "Silero VAD failed; retaining this window.", failure)
            true
        }
    }

    override fun close() {
        sampleRateTensor.close()
        stateTensor.close()
        inputTensor.close()
        session.close()
    }

    companion object {
        const val MODEL_ASSET_PATH = "silero/silero_vad.onnx"
        const val MODEL_VERSION = "6.2.1"
        const val MODEL_REVISION = "7e30209a3e901f9842f81b225f3e93d8199902b1"
        const val MODEL_SHA256 =
            "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

        private const val TAG = "SileroVAD"
        private const val INPUT_NAME = "input"
        private const val STATE_NAME = "state"
        private const val SAMPLE_RATE_NAME = "sr"
        private const val OUTPUT_NAME = "output"
        private const val NEXT_STATE_NAME = "stateN"
        private const val FRAME_SAMPLES = 512
        private const val CONTEXT_SAMPLES = 64
        private const val INPUT_SAMPLES = FRAME_SAMPLES + CONTEXT_SAMPLES
        private const val STATE_FLOAT_COUNT = 2 * 1 * 128

        fun open(context: Context): SileroVoiceActivityDetector {
            val modelBytes = context.assets.open(MODEL_ASSET_PATH).use { it.readBytes() }
            val environment = OrtEnvironment.getEnvironment()
            val session = OrtSession.SessionOptions().use { options ->
                options.setInterOpNumThreads(1)
                options.setIntraOpNumThreads(1)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                environment.createSession(modelBytes, options)
            }
            try {
                check(
                    session.inputNames == setOf(INPUT_NAME, STATE_NAME, SAMPLE_RATE_NAME),
                ) {
                    "Unexpected Silero inputs: ${session.inputNames.sorted()}."
                }
                check(
                    session.outputNames == setOf(OUTPUT_NAME, NEXT_STATE_NAME),
                ) {
                    "Unexpected Silero outputs: ${session.outputNames.sorted()}."
                }

                val inputBuffer = directFloatBuffer(INPUT_SAMPLES)
                val stateBuffer = directFloatBuffer(STATE_FLOAT_COUNT)
                val sampleRateBuffer = directLongBuffer(1).apply {
                    put(WhisperFeatureExtractor.SAMPLE_RATE.toLong())
                    rewind()
                }
                val inputTensor = OnnxTensor.createTensor(
                    environment,
                    inputBuffer,
                    longArrayOf(1, INPUT_SAMPLES.toLong()),
                )
                val stateTensor = OnnxTensor.createTensor(
                    environment,
                    stateBuffer,
                    longArrayOf(2, 1, 128),
                )
                val sampleRateTensor = OnnxTensor.createTensor(
                    environment,
                    sampleRateBuffer,
                    longArrayOf(1),
                )
                return SileroVoiceActivityDetector(
                    session = session,
                    inputTensor = inputTensor,
                    stateTensor = stateTensor,
                    sampleRateTensor = sampleRateTensor,
                    inputBuffer = inputBuffer,
                    stateBuffer = stateBuffer,
                )
            } catch (failure: Throwable) {
                session.close()
                throw failure
            }
        }

        private fun directFloatBuffer(size: Int): FloatBuffer =
            ByteBuffer.allocateDirect(size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()

        private fun directLongBuffer(size: Int): LongBuffer =
            ByteBuffer.allocateDirect(size * Long.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asLongBuffer()
    }
}

/**
 * Conservative gate over Silero probabilities. Three positive 32 ms frames
 * are sufficient, so short or quiet utterances are retained rather than
 * requiring the upstream default 250 ms minimum speech duration.
 */
internal class SileroSpeechGate(
    private val threshold: Float = 0.5f,
    private val minimumConsecutiveFrames: Int = 3,
) {
    private var consecutiveFrames = 0

    init {
        require(threshold in 0f..1f)
        require(minimumConsecutiveFrames > 0)
    }

    fun observe(speechProbability: Float): Boolean {
        consecutiveFrames = if (speechProbability >= threshold) {
            consecutiveFrames + 1
        } else {
            0
        }
        return consecutiveFrames >= minimumConsecutiveFrames
    }
}
