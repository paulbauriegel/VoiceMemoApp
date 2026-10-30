package com.example.voicememo.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/** Produces unit-length sentence embeddings for semantic transcript search. */
interface SentenceEmbedder : AutoCloseable {
    val modelId: String

    fun embedPassage(text: String): FloatArray

    fun embedQuery(text: String): FloatArray
}

/**
 * Offline multilingual-e5-small (int8, 384 dimensions) on the ONNX Runtime CPU
 * provider. e5 is trained with asymmetric `query: ` / `passage: ` prefixes;
 * vectors are attention-masked mean pools, L2 normalized, so a dot product is
 * cosine similarity. Inputs are run one at a time without padding, because
 * the dynamically quantized graph computes activation scales per batch.
 */
class MultilingualE5Embedder private constructor(
    private val session: OrtSession,
    private val tokenizer: SentencePieceUnigramTokenizer,
) : SentenceEmbedder {
    override val modelId: String = MODEL_ID

    override fun embedPassage(text: String): FloatArray = embed(PASSAGE_PREFIX + text)

    override fun embedQuery(text: String): FloatArray = embed(QUERY_PREFIX + text)

    @Synchronized
    private fun embed(text: String): FloatArray {
        val ids = tokenizer.encode(text)
        val shape = longArrayOf(1, ids.size.toLong())
        val environment = OrtEnvironment.getEnvironment()
        val mask = LongArray(ids.size) { 1L }
        val tokenTypes = LongArray(ids.size)
        OnnxTensor.createTensor(environment, java.nio.LongBuffer.wrap(ids), shape).use { idTensor ->
            OnnxTensor.createTensor(environment, java.nio.LongBuffer.wrap(mask), shape).use { maskTensor ->
                OnnxTensor.createTensor(environment, java.nio.LongBuffer.wrap(tokenTypes), shape)
                    .use { typeTensor ->
                        session.run(
                            mapOf(
                                "input_ids" to idTensor,
                                "attention_mask" to maskTensor,
                                "token_type_ids" to typeTensor,
                            ),
                        ).use { result ->
                            val hidden = (result[0] as OnnxTensor).floatBuffer
                            return meanPoolNormalized(hidden, ids.size)
                        }
                    }
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MODEL_ID = "multilingual-e5-small-qint8@614241f6"
        const val MODEL_ASSET_PATH = "embedding/multilingual_e5_small_qint8.onnx"
        const val TOKENIZER_ASSET_PATH = "embedding/multilingual_e5_small_sentencepiece.model"
        const val MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        const val DIMENSIONS = 384
        private const val QUERY_PREFIX = "query: "
        private const val PASSAGE_PREFIX = "passage: "

        fun open(context: Context): MultilingualE5Embedder {
            val tokenizer = context.assets.open(TOKENIZER_ASSET_PATH)
                .use(SentencePieceUnigramTokenizer::load)
            // The asset is stored uncompressed, so it can be memory-mapped
            // instead of copying 118 MB through the Java heap.
            val modelBuffer = context.assets.openFd(MODEL_ASSET_PATH).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                    channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength,
                    )
                }
            }
            val session = OrtSession.SessionOptions().use { options ->
                options.setInterOpNumThreads(1)
                options.setIntraOpNumThreads(INTRA_OP_THREADS)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                OrtEnvironment.getEnvironment().createSession(modelBuffer, options)
            }
            try {
                check(
                    session.inputNames == setOf("input_ids", "attention_mask", "token_type_ids"),
                ) {
                    "Unexpected e5 inputs: ${session.inputNames.sorted()}."
                }
            } catch (failure: Throwable) {
                session.close()
                throw failure
            }
            return MultilingualE5Embedder(session, tokenizer)
        }

        private const val INTRA_OP_THREADS = 2

        internal fun meanPoolNormalized(
            hidden: java.nio.FloatBuffer,
            tokenCount: Int,
        ): FloatArray {
            val vector = FloatArray(DIMENSIONS)
            for (token in 0 until tokenCount) {
                val base = token * DIMENSIONS
                for (dimension in 0 until DIMENSIONS) {
                    vector[dimension] += hidden.get(base + dimension)
                }
            }
            var norm = 0.0
            for (value in vector) norm += value * value
            val scale = if (norm > 0.0) (1.0 / sqrt(norm)).toFloat() else 0f
            for (dimension in vector.indices) vector[dimension] *= scale
            return vector
        }
    }
}
