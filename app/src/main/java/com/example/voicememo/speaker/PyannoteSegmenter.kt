package com.example.voicememo.speaker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

/**
 * Offline pyannote segmentation-3.0 through ONNX Runtime. One call covers a
 * fixed 10-second window and says, for each of 589 frames, which of up to three
 * window-local speakers are talking. Speaker numbers are only meaningful inside
 * one window.
 *
 * ONNX Runtime with two threads is 2.6x faster than LiteRT for this model on a
 * Pixel 7 Pro (see docs/transcription-benchmarks.md); four threads are slower.
 */
internal class PyannoteSegmenter private constructor(
    private val session: OrtSession,
    private val inputTensor: OnnxTensor,
    private val inputBuffer: FloatBuffer,
) : AutoCloseable {

    /**
     * Log-probabilities `[FRAMES, SpeakerPowerset.CLASSES]` for [WINDOW_SAMPLES]
     * samples of 16 kHz audio starting at [offset]; missing samples are silence.
     */
    @Synchronized
    fun logProbabilities(audio16Khz: FloatArray, offset: Int = 0): FloatArray {
        require(offset >= 0)
        val available = (audio16Khz.size - offset).coerceIn(0, WINDOW_SAMPLES)
        inputBuffer.clear()
        if (available > 0) inputBuffer.put(audio16Khz, offset, available)
        repeat(WINDOW_SAMPLES - available) { inputBuffer.put(0f) }
        inputBuffer.rewind()
        return session.run(mapOf(INPUT_NAME to inputTensor)).use { result ->
            val output = (result[0] as OnnxTensor).floatBuffer
            check(output.remaining() == FRAMES * SpeakerPowerset.CLASSES) {
                "Unexpected segmentation output size ${output.remaining()}."
            }
            FloatArray(output.remaining()).also(output::get)
        }
    }

    @Synchronized
    override fun close() {
        inputTensor.close()
        session.close()
    }

    companion object {
        const val MODEL_ASSET_PATH = "speaker/pyannote_segmentation_3_0.onnx"
        const val MODEL_VERSION = "3.0"
        const val MODEL_SHA256 =
            "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
        const val SAMPLE_RATE = 16_000
        const val WINDOW_SAMPLES = 160_000
        const val FRAMES = 589
        /** Frame f covers samples from `f * FRAME_SHIFT_SAMPLES`, RECEPTIVE_FIELD_SAMPLES long. */
        const val FRAME_SHIFT_SAMPLES = 270
        const val RECEPTIVE_FIELD_SAMPLES = 991
        const val DEFAULT_THREADS = 2
        private const val INPUT_NAME = "x"
        private const val OUTPUT_NAME = "y"

        fun open(context: Context, threads: Int = DEFAULT_THREADS): PyannoteSegmenter =
            create(threads) { options ->
                OrtEnvironment.getEnvironment()
                    .createSession(SpeakerModelAssets.map(context, MODEL_ASSET_PATH), options)
            }

        fun open(modelFile: File, threads: Int = DEFAULT_THREADS): PyannoteSegmenter =
            create(threads) { options ->
                OrtEnvironment.getEnvironment().createSession(modelFile.absolutePath, options)
            }

        private fun create(
            threads: Int,
            createSession: (OrtSession.SessionOptions) -> OrtSession,
        ): PyannoteSegmenter {
            val session = OrtSession.SessionOptions().use { options ->
                options.setInterOpNumThreads(1)
                options.setIntraOpNumThreads(threads)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                createSession(options)
            }
            try {
                check(session.inputNames == setOf(INPUT_NAME)) {
                    "Unexpected segmentation inputs: ${session.inputNames.sorted()}."
                }
                check(session.outputNames == setOf(OUTPUT_NAME)) {
                    "Unexpected segmentation outputs: ${session.outputNames.sorted()}."
                }
                val buffer = ByteBuffer.allocateDirect(WINDOW_SAMPLES * Float.SIZE_BYTES)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                val tensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    buffer,
                    longArrayOf(1, 1, WINDOW_SAMPLES.toLong()),
                )
                return PyannoteSegmenter(session, tensor, buffer)
            } catch (failure: Throwable) {
                session.close()
                throw failure
            }
        }
    }
}

internal object SpeakerModelAssets {
    /** The assets are stored uncompressed, so they are memory-mapped, not copied. */
    fun map(context: Context, assetPath: String): ByteBuffer =
        context.assets.openFd(assetPath).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength,
                )
            }
        }
}
