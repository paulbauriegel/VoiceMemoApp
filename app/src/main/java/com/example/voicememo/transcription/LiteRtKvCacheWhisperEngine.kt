package com.example.voicememo.transcription

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Research-only fixed-shape KV-cache decoder benchmark. */
internal class LiteRtKvCacheWhisperEngine private constructor(
    private val encoderModel: CompiledModel,
    private val crossCacheModel: CompiledModel,
    private val stepModel: CompiledModel,
    private val projection: ActiveRowProjection,
    private val tokenizer: WhisperTokenizer,
    private val generationConfig: WhisperGenerationConfig,
    private val featureExtractor: WhisperFeatureExtractor,
    private val encoderFeatures: TensorBuffer,
    private val encoderState: TensorBuffer,
    private val crossEncoderState: TensorBuffer,
    private val crossOutputs: Map<String, TensorBuffer>,
    private val stepToken: TensorBuffer,
    private val stepPosition: TensorBuffer,
    private val stepCrossInputs: Map<String, TensorBuffer>,
    private val stepSelfInputs: Map<String, TensorBuffer>,
    private val stepSelfOutputs: Map<String, TensorBuffer>,
    private val stepHidden: TensorBuffer,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "KV-cache Whisper engine is closed." }
        require(language is WhisperLanguage.Forced) {
            "The KV-cache benchmark requires a forced language."
        }
        require(audio.size <= WhisperFeatureExtractor.MAX_SAMPLES)
        val started = System.nanoTime()
        checkCancellation(shouldCancel)
        encoderFeatures.writeFloat(featureExtractor.extract(audio))
        encoderModel.run(
            mapOf(ENCODE_INPUT to encoderFeatures),
            mapOf(ENCODE_OUTPUT to encoderState),
            ENCODE_SIGNATURE,
        )
        val state = encoderState.readFloat()
        require(state.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
        initializeCaches(state, shouldCancel)

        val prompt = listOf(
            generationConfig.decoderStartTokenId,
            generationConfig.languageToken(language.code),
            generationConfig.transcribeTokenId,
            generationConfig.noTimestampsTokenId,
        )
        var hidden = FloatArray(WhisperSplitDecoderTensors.ENCODER_WIDTH)
        prompt.forEachIndexed { position, token ->
            hidden = cachedStep(token, position, shouldCancel)
        }

        val generated = mutableListOf<Int>()
        var reachedEos = false
        val generationLimit = WhisperSplitDecoderTensors.TARGET_LENGTH - prompt.size
        onTokenProgress(0, generationLimit)
        while (generated.size < generationLimit) {
            checkCancellation(shouldCancel)
            val logits = projection.project(hidden, shouldCancel)
            val ranking = WhisperTeacherForcedTokenRanking.forCpu(
                logits = logits,
                generationConfig = generationConfig,
                firstGeneratedToken = generated.isEmpty(),
            )
            val next = stableCachedTop1(ranking)?.tokenId
                ?: throw WhisperModelContractException("KV-cache decoder produced no token.")
            if (next == generationConfig.eosTokenId) {
                reachedEos = true
                break
            }
            generated += next
            onTokenProgress(generated.size, generationLimit)
            if (generated.size < generationLimit) {
                hidden = cachedStep(
                    token = next,
                    position = prompt.size + generated.lastIndex,
                    shouldCancel = shouldCancel,
                )
            }
        }
        val truncated = !reachedEos && generated.size >= generationLimit
        WhisperChunkResult(
            tokens = generated,
            text = tokenizer.decode(generated).trim(),
            detectedLanguage = language.code,
            inferenceMillis = (System.nanoTime() - started) / 1_000_000L,
            truncated = truncated,
            warnings = if (truncated) listOf("Decoder reached its 128-token limit before EOS.") else emptyList(),
        )
    }

    /**
     * Computes cached-decoder rows while forcing the exact supplied token path.
     * This avoids autoregressive cascade and is available only to diagnostics.
     */
    internal fun teacherForcedLogitsRows(
        preparedEncoderState: FloatArray,
        language: WhisperLanguage.Forced,
        referenceTokens: List<Int>,
        shouldCancel: () -> Boolean = { false },
    ): List<WhisperTeacherForcedLogitsRow> = synchronized(inferenceLock) {
        check(!closed.get()) { "KV-cache Whisper engine is closed." }
        require(preparedEncoderState.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
        require(referenceTokens.size <= WhisperSplitDecoderTensors.TARGET_LENGTH - 4)
        initializeCaches(preparedEncoderState, shouldCancel)

        val prompt = listOf(
            generationConfig.decoderStartTokenId,
            generationConfig.languageToken(language.code),
            generationConfig.transcribeTokenId,
            generationConfig.noTimestampsTokenId,
        )
        var hidden = FloatArray(WhisperSplitDecoderTensors.ENCODER_WIDTH)
        prompt.forEachIndexed { position, token ->
            hidden = cachedStep(token, position, shouldCancel)
        }
        referenceTokens.mapIndexed { index, token ->
            checkCancellation(shouldCancel)
            val row = WhisperTeacherForcedTokenRanking.forCpu(
                logits = projection.project(hidden, shouldCancel),
                generationConfig = generationConfig,
                firstGeneratedToken = index == 0,
            )
            if (index < referenceTokens.lastIndex) {
                hidden = cachedStep(
                    token = token,
                    position = prompt.size + index,
                    shouldCancel = shouldCancel,
                )
            }
            row
        }
    }

    private fun initializeCaches(
        preparedEncoderState: FloatArray,
        shouldCancel: () -> Boolean,
    ) {
        require(preparedEncoderState.size == WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
        checkCancellation(shouldCancel)
        crossEncoderState.writeFloat(preparedEncoderState)
        crossCacheModel.run(
            mapOf(CROSS_INPUT to crossEncoderState),
            crossOutputs,
            CROSS_SIGNATURE,
        )
        CROSS_CACHE_NAMES.forEach { name ->
            stepCrossInputs.getValue(name).writeFloat(crossOutputs.getValue(name).readFloat())
        }
        val zeroCache = FloatArray(SELF_CACHE_FLOATS)
        SELF_CACHE_NAMES.forEach { name -> stepSelfInputs.getValue(name).writeFloat(zeroCache) }
    }

    private fun cachedStep(
        token: Int,
        position: Int,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        checkCancellation(shouldCancel)
        stepToken.writeInt(intArrayOf(token))
        stepPosition.writeInt(intArrayOf(position))
        val inputs = HashMap<String, TensorBuffer>(18).apply {
            put(STEP_TOKEN, stepToken)
            put(STEP_POSITION, stepPosition)
            putAll(stepCrossInputs)
            putAll(stepSelfInputs)
        }
        val outputs = HashMap<String, TensorBuffer>(9).apply {
            put(STEP_HIDDEN, stepHidden)
            putAll(stepSelfOutputs)
        }
        stepModel.run(inputs, outputs, STEP_SIGNATURE)
        SELF_CACHE_NAMES.forEach { name ->
            stepSelfInputs.getValue(name).writeFloat(stepSelfOutputs.getValue(name).readFloat())
        }
        return stepHidden.readFloat().also {
            require(it.size == WhisperSplitDecoderTensors.ENCODER_WIDTH)
        }
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(tokens)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            runCatching(projection::close)
            allBuffers().asReversed().forEach { buffer -> runCatching(buffer::close) }
            runCatching(stepModel::close)
            runCatching(crossCacheModel::close)
            encoderModel.close()
        }
    }

    private fun allBuffers(): List<TensorBuffer> = buildList {
        add(encoderFeatures)
        add(encoderState)
        add(crossEncoderState)
        addAll(crossOutputs.values)
        add(stepToken)
        add(stepPosition)
        addAll(stepCrossInputs.values)
        addAll(stepSelfInputs.values)
        addAll(stepSelfOutputs.values)
        add(stepHidden)
    }

    private fun checkCancellation(shouldCancel: () -> Boolean) {
        if (closed.get() || shouldCancel()) throw WhisperCancelledException()
    }

    companion object {
        private const val ENCODE_SIGNATURE = "encode"
        private const val ENCODE_INPUT = "args_0"
        private const val ENCODE_OUTPUT = "output_0"
        private const val CROSS_SIGNATURE = "initialize"
        private const val CROSS_INPUT = "encoder_state"
        private const val STEP_SIGNATURE = "step"
        private const val STEP_TOKEN = "token"
        private const val STEP_POSITION = "position"
        private const val STEP_HIDDEN = "hidden"
        // The full F32_F16 decoder can collapse nearly equal logits to an exact
        // FP16-grid tie. Preserve its ascending-token tie behavior when the
        // single-row cached graph separates the same pair by at most 1/32.
        internal const val F32_F16_TIE_WINDOW = 1.0f / 32.0f
        private const val SELF_CACHE_FLOATS = 20 * 128 * 64
        private val SELF_CACHE_NAMES = cacheNames("self")
        private val CROSS_CACHE_NAMES = cacheNames("cross")

        private fun cacheNames(kind: String): List<String> = buildList {
            repeat(4) { layer ->
                add("layer_${layer}_${kind}_key")
                add("layer_${layer}_${kind}_value")
            }
        }

        internal fun stableCachedTop1(
            ranking: WhisperTeacherForcedLogitsRow,
        ): WhisperLogitCandidate? {
            val top1 = ranking.selectableTop1 ?: return null
            val top2 = ranking.selectableTop2 ?: return top1
            val margin = ranking.selectableMargin ?: return top1
            return if (margin <= F32_F16_TIE_WINDOW && top2.tokenId < top1.tokenId) {
                top2
            } else {
                top1
            }
        }

        fun open(
            context: Context,
            encoderModelPath: String,
            crossCacheModelPath: String,
            stepModelPath: String,
            projectionModelPath: String,
            encoderPrecision: CompiledModel.GpuOptions.Precision,
            decoderPrecision: CompiledModel.GpuOptions.Precision,
        ): LiteRtKvCacheWhisperEngine {
            fun options(precision: CompiledModel.GpuOptions.Precision) =
                CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = true,
                        allowSrcQuantizedFcConvOps = false,
                        precision = precision,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    )
                }
            val encoder = CompiledModel.create(encoderModelPath, options(encoderPrecision))
            var cross: CompiledModel? = null
            var step: CompiledModel? = null
            var projection: ActiveRowProjection? = null
            val buffers = mutableListOf<TensorBuffer>()
            try {
                cross = CompiledModel.create(crossCacheModelPath, options(decoderPrecision))
                step = CompiledModel.create(stepModelPath, options(decoderPrecision))
                projection = ActiveRowProjection.openFile(projectionModelPath, 3)
                fun input(model: CompiledModel, name: String, signature: String) =
                    model.createInputBuffer(name, signature).also(buffers::add)
                fun output(model: CompiledModel, name: String, signature: String) =
                    model.createOutputBuffer(name, signature).also(buffers::add)
                val assets = context.applicationContext.assets
                val tokenizer = assets.open(WhisperAssets.TOKENIZER).use(WhisperTokenizer::load)
                val config = assets.open(WhisperAssets.GENERATION_CONFIG).use(WhisperGenerationConfig::load)
                return LiteRtKvCacheWhisperEngine(
                    encoderModel = encoder,
                    crossCacheModel = cross,
                    stepModel = step,
                    projection = projection,
                    tokenizer = tokenizer,
                    generationConfig = config,
                    featureExtractor = WhisperFeatureExtractor(),
                    encoderFeatures = input(encoder, ENCODE_INPUT, ENCODE_SIGNATURE),
                    encoderState = output(encoder, ENCODE_OUTPUT, ENCODE_SIGNATURE),
                    crossEncoderState = input(cross, CROSS_INPUT, CROSS_SIGNATURE),
                    crossOutputs = CROSS_CACHE_NAMES.associateWith { output(cross, it, CROSS_SIGNATURE) },
                    stepToken = input(step, STEP_TOKEN, STEP_SIGNATURE),
                    stepPosition = input(step, STEP_POSITION, STEP_SIGNATURE),
                    stepCrossInputs = CROSS_CACHE_NAMES.associateWith { input(step, it, STEP_SIGNATURE) },
                    stepSelfInputs = SELF_CACHE_NAMES.associateWith { input(step, it, STEP_SIGNATURE) },
                    stepSelfOutputs = SELF_CACHE_NAMES.associateWith { output(step, it, STEP_SIGNATURE) },
                    stepHidden = output(step, STEP_HIDDEN, STEP_SIGNATURE),
                )
            } catch (error: Throwable) {
                runCatching { projection?.close() }
                buffers.asReversed().forEach { buffer -> runCatching(buffer::close) }
                runCatching { step?.close() }
                runCatching { cross?.close() }
                encoder.close()
                throw error
            }
        }
    }
}
