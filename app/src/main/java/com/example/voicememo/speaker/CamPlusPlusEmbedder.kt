package com.example.voicememo.speaker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Offline CAM++ voice model through ONNX Runtime: turns a stretch of speech
 * into a voiceprint. Voiceprints of the same person point in a similar
 * direction (cosine similarity).
 *
 * The bundled model is 3D-Speaker CAM++ trained on Chinese and English
 * (`speech_campplus_sv_zh_en_16k-common_advanced`, 192 numbers). On the
 * hand-marked recording it separates voices better than WeSpeaker CAM++
 * trained on VoxCeleb, which was bundled before (see
 * docs/transcription-benchmarks.md).
 *
 * Two ONNX Runtime 1.26 defects shape how the model is run. Re-run
 * scripts/run-speaker-chain-benchmark.py before changing either or the ONNX
 * Runtime version:
 *  - CAM++ pools over segments of 200 feature frames (2 s). 1.26 averages a
 *    partial last segment differently from 1.32, at every optimisation level,
 *    so voiceprints of less than about 4 s come out different. The feature
 *    frames are therefore repeated from the start up to the next multiple of
 *    [FRAME_PADDING], which gives the same result on both versions.
 *  - The session is created with graph optimisation off. 1.26 computes the
 *    WeSpeaker CAM++ export wrongly at every level above NO_OPT; the bundled
 *    model is not affected, and NO_OPT costs no measurable speed.
 */
internal class CamPlusPlusEmbedder private constructor(
    private val session: OrtSession,
    /** Voiceprint length; [DIMENSIONS] for the bundled model. */
    val dimensions: Int,
    private val inputName: String,
    /** Multiplier for samples in -1..1 before the features are computed. */
    private val sampleScale: Float,
    /** Feature frames are repeated up to the next multiple of this; 0 leaves them as they are. */
    private val framePadding: Int,
) : AutoCloseable {
    private val fbank = KaldiFbank()

    /**
     * Unit-length voiceprint of `audio16Khz[offset, offset + length)`, or null
     * when it is shorter than [MIN_SAMPLES].
     */
    @Synchronized
    fun embed(audio16Khz: FloatArray, offset: Int = 0, length: Int = audio16Khz.size - offset): FloatArray? {
        if (length < MIN_SAMPLES) return null
        val computed = fbank.compute(audio16Khz, offset, length, scale = sampleScale, subtractMean = true)
        val features = padFrames(computed, framePadding)
        val frames = features.size / KaldiFbank.MEL_BINS
        val buffer = ByteBuffer.allocateDirect(features.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(features)
        buffer.rewind()
        val environment = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(
            environment,
            buffer,
            longArrayOf(1, frames.toLong(), KaldiFbank.MEL_BINS.toLong()),
        ).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val output = (result[0] as OnnxTensor).floatBuffer
                check(output.remaining() == dimensions) {
                    "Unexpected voiceprint size ${output.remaining()}."
                }
                normalise(FloatArray(dimensions).also(output::get))
            }
        }
    }

    @Synchronized
    override fun close() {
        session.close()
    }

    companion object {
        const val MODEL_ID = "3dspeaker-campplus-zh-en-advanced"
        const val MODEL_ASSET_PATH = "speaker/3dspeaker_campplus_zh_en_advanced.onnx"
        const val MODEL_SHA256 =
            "53cab20ba8f2eb2f6e2d89ecc18e88ef3222169b61b4419cbc13666744a4574b"
        const val DIMENSIONS = 192
        /** A quarter of a second; the labelling chain was validated with this minimum. */
        const val MIN_SAMPLES = 4_000
        const val DEFAULT_THREADS = 2
        /** 3D-Speaker models take samples in -1..1; WeSpeaker models need [WESPEAKER_SAMPLE_SCALE]. */
        const val SAMPLE_SCALE = 1f
        const val WESPEAKER_SAMPLE_SCALE = 32_768f
        /** Two seconds of feature frames: one CAM++ pooling segment after its 2x subsampling. */
        const val FRAME_PADDING = 200
        private const val INPUT_NAME = "x"
        private const val OUTPUT_NAME = "embedding"

        fun open(context: Context, threads: Int = DEFAULT_THREADS): CamPlusPlusEmbedder =
            create(threads, graphOptimisation = false, anyNames = false, SAMPLE_SCALE, FRAME_PADDING) { options ->
                OrtEnvironment.getEnvironment()
                    .createSession(SpeakerModelAssets.map(context, MODEL_ASSET_PATH), options)
            }.also { embedder ->
                if (embedder.dimensions != DIMENSIONS) {
                    embedder.close()
                    error("Unexpected voiceprint size ${embedder.dimensions}.")
                }
            }

        /**
         * Any voice model file with one fbank input `[1, frames, 80]` and one
         * voiceprint output; for benchmarks. The defaults are those of the
         * bundled model.
         */
        fun open(
            modelFile: File,
            threads: Int = DEFAULT_THREADS,
            graphOptimisation: Boolean = false,
            sampleScale: Float = SAMPLE_SCALE,
            framePadding: Int = FRAME_PADDING,
        ): CamPlusPlusEmbedder =
            create(threads, graphOptimisation, anyNames = true, sampleScale, framePadding) { options ->
                OrtEnvironment.getEnvironment().createSession(modelFile.absolutePath, options)
            }

        fun normalise(vector: FloatArray): FloatArray {
            var sum = 0.0
            for (value in vector) sum += value * value
            val norm = sqrt(sum).toFloat()
            if (norm > 0f) for (index in vector.indices) vector[index] /= norm
            return vector
        }

        /**
         * [features] (row-major, [KaldiFbank.MEL_BINS] per frame) with its frames
         * repeated from the start up to the next multiple of [multiple] frames.
         */
        internal fun padFrames(features: FloatArray, multiple: Int): FloatArray {
            if (multiple <= 0 || features.isEmpty()) return features
            val frames = features.size / KaldiFbank.MEL_BINS
            val padded = (frames + multiple - 1) / multiple * multiple
            if (padded == frames) return features
            return FloatArray(padded * KaldiFbank.MEL_BINS) { features[it % features.size] }
        }

        private fun create(
            threads: Int,
            graphOptimisation: Boolean,
            anyNames: Boolean,
            sampleScale: Float,
            framePadding: Int,
            createSession: (OrtSession.SessionOptions) -> OrtSession,
        ): CamPlusPlusEmbedder {
            val session = OrtSession.SessionOptions().use { options ->
                options.setInterOpNumThreads(1)
                options.setIntraOpNumThreads(threads)
                options.setOptimizationLevel(
                    if (graphOptimisation) {
                        OrtSession.SessionOptions.OptLevel.ALL_OPT
                    } else {
                        OrtSession.SessionOptions.OptLevel.NO_OPT
                    },
                )
                createSession(options)
            }
            try {
                check(if (anyNames) session.inputNames.size == 1 else session.inputNames == setOf(INPUT_NAME)) {
                    "Unexpected voice model inputs: ${session.inputNames.sorted()}."
                }
                check(if (anyNames) session.outputNames.size == 1 else session.outputNames == setOf(OUTPUT_NAME)) {
                    "Unexpected voice model outputs: ${session.outputNames.sorted()}."
                }
                val output = session.outputNames.single()
                val shape = (session.outputInfo.getValue(output).info as TensorInfo).shape
                return CamPlusPlusEmbedder(
                    session,
                    shape.last().toInt(),
                    session.inputNames.single(),
                    sampleScale,
                    framePadding,
                )
            } catch (failure: Throwable) {
                session.close()
                throw failure
            }
        }
    }
}
