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
    /** Exactly one of [projection] and [stepLogits] is non-null. */
    private val projection: ActiveRowProjection?,
    private val stepLogits: TensorBuffer?,
    private val tokenizer: WhisperTokenizer,
    private val generationConfig: WhisperGenerationConfig,
    private val featureExtractor: WhisperFeatureExtractor,
    private val timestampsEnabled: Boolean,
    private val encoderFeatures: TensorBuffer,
    private val encoderState: TensorBuffer,
    private val crossEncoderState: TensorBuffer,
    private val crossOutputs: Map<String, TensorBuffer>,
    private val stepToken: TensorBuffer,
    private val stepPosition: TensorBuffer,
    private val stepCrossInputs: Map<String, TensorBuffer>,
    private val stepSelfCachesA: Map<String, TensorBuffer>,
    private val stepSelfCachesB: Map<String, TensorBuffer>,
    private val stepHidden: TensorBuffer,
) : WhisperEngine {
    private val closed = AtomicBoolean(false)
    private val inferenceLock = Any()

    // Self-attention caches ping-pong between the A and B buffer sets: each
    // step reads the current set and writes the complete updated cache to the
    // other, so no per-token GPU→CPU→GPU copy is needed. The step graph emits
    // the full cache tensor (input cache with the current row replaced), which
    // is the same property the previous wholesale copy relied on.
    private var selfCacheFlip = false
    private val zeroSelfCache = FloatArray(SELF_CACHE_FLOATS)

    init {
        require((projection == null) != (stepLogits == null)) {
            "Exactly one of the separate projection and the fused logits output must be present."
        }
    }

    /**
     * Logits for the hidden row produced by the MOST RECENT [cachedStep] call.
     * Fused models read the step's own logits output, which stays valid only
     * until the next step runs; the separate CPU projection recomputes from
     * the hidden row. Both call sites consume logits before stepping again.
     */
    private fun logitsForLastStep(hidden: FloatArray, shouldCancel: () -> Boolean): FloatArray =
        stepLogits?.readFloat()?.also {
            require(it.size == WhisperSplitDecoderTensors.VOCABULARY_SIZE)
        } ?: checkNotNull(projection).project(hidden, shouldCancel)

    override fun transcribeChunk(
        audio: FloatArray,
        language: WhisperLanguage,
        shouldCancel: () -> Boolean,
        onTokenProgress: (generated: Int, limit: Int) -> Unit,
    ): WhisperChunkResult = synchronized(inferenceLock) {
        check(!closed.get()) { "KV-cache Whisper engine is closed." }
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

        // Position 0 is always start-of-transcript. For automatic language
        // detection, its logits already rank the language tokens, and the
        // position-0 cache entry stays valid, so detection costs one
        // projection and no extra decoder steps.
        var hidden = cachedStep(generationConfig.decoderStartTokenId, 0, shouldCancel)
        val languageToken: Int
        val resolvedLanguage: String
        when (language) {
            is WhisperLanguage.Forced -> {
                languageToken = generationConfig.languageToken(language.code)
                resolvedLanguage = language.code
            }
            is WhisperLanguage.Auto -> {
                val logits = logitsForLastStep(hidden, shouldCancel)
                languageToken = WhisperGpuGreedyTokenSelector.selectLanguageToken(
                    logits = logits,
                    position = 0,
                    vocabularySize = WhisperSplitDecoderTensors.VOCABULARY_SIZE,
                    languageTokenIds = generationConfig.idToLanguage.keys,
                )
                resolvedLanguage = generationConfig.idToLanguage[languageToken]
                    ?: throw WhisperModelContractException(
                        "Decoder selected unknown language token $languageToken.",
                    )
            }
        }
        buildList {
            add(languageToken)
            add(generationConfig.transcribeTokenId)
            if (!timestampsEnabled) add(generationConfig.noTimestampsTokenId)
        }.forEachIndexed { index, token ->
            hidden = cachedStep(token, 1 + index, shouldCancel)
        }
        val promptLength = if (timestampsEnabled) 3 else 4

        val generated = mutableListOf<Int>()
        var reachedEos = false
        val generationLimit = WhisperSplitDecoderTensors.TARGET_LENGTH - promptLength
        onTokenProgress(0, generationLimit)
        while (generated.size < generationLimit) {
            checkCancellation(shouldCancel)
            val logits = logitsForLastStep(hidden, shouldCancel)
            val next = if (timestampsEnabled) {
                // Timestamp mode uses the full engine's ApplyTimestampRules
                // selector unchanged. The F32_F16 near-tie stabilizer below is
                // specific to the plain-text ranking and is intentionally not
                // grafted into the timestamp grammar.
                WhisperTimestampTokenSelector.select(
                    vocabularySize = WhisperSplitDecoderTensors.VOCABULARY_SIZE,
                    generatedTokens = generated,
                    config = generationConfig,
                    firstGeneratedToken = generated.isEmpty(),
                    logitAt = { tokenId -> logits[tokenId] },
                )
            } else {
                val ranking = WhisperTeacherForcedTokenRanking.forCpu(
                    logits = logits,
                    generationConfig = generationConfig,
                    firstGeneratedToken = generated.isEmpty(),
                )
                stableCachedTop1(ranking)?.tokenId
                    ?: throw WhisperModelContractException("KV-cache decoder produced no token.")
            }
            if (next == generationConfig.eosTokenId) {
                reachedEos = true
                break
            }
            generated += next
            onTokenProgress(generated.size, generationLimit)
            if (generated.size < generationLimit) {
                hidden = cachedStep(
                    token = next,
                    position = promptLength + generated.lastIndex,
                    shouldCancel = shouldCancel,
                )
            }
        }
        val truncated = !reachedEos && generated.size >= generationLimit
        WhisperChunkResult(
            tokens = generated,
            text = decodeTokens(generated).trim(),
            detectedLanguage = resolvedLanguage,
            inferenceMillis = (System.nanoTime() - started) / 1_000_000L,
            truncated = truncated,
            warnings = if (truncated) listOf("Decoder reached its 128-token limit before EOS.") else emptyList(),
            timedSegments = if (timestampsEnabled) {
                WhisperTimestampTokens.parseSegments(
                    tokens = generated,
                    timestampBeginTokenId = generationConfig.timestampBeginTokenId,
                    tokenizer = tokenizer,
                    audioDurationMillis = audio.size * 1000L / 16_000L,
                )
            } else {
                emptyList()
            },
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
                logits = logitsForLastStep(hidden, shouldCancel),
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
        // One host round-trip per window: TensorBuffers are bound to the model
        // that created them (separate CL contexts), so the cross model's
        // outputs cannot be passed to the step model directly — attempting
        // that fails CompiledModel.run. This copy is per-window, not
        // per-token, so it is not on the generation hot path.
        CROSS_CACHE_NAMES.forEach { name ->
            stepCrossInputs.getValue(name).writeFloat(crossOutputs.getValue(name).readFloat())
        }
        selfCacheFlip = false
        SELF_CACHE_NAMES.forEach { name ->
            stepSelfCachesA.getValue(name).writeFloat(zeroSelfCache)
        }
    }

    private fun cachedStep(
        token: Int,
        position: Int,
        shouldCancel: () -> Boolean,
    ): FloatArray {
        checkCancellation(shouldCancel)
        stepToken.writeInt(intArrayOf(token))
        stepPosition.writeInt(intArrayOf(position))
        val inputSelf = if (selfCacheFlip) stepSelfCachesB else stepSelfCachesA
        val outputSelf = if (selfCacheFlip) stepSelfCachesA else stepSelfCachesB
        val inputs = HashMap<String, TensorBuffer>(18).apply {
            put(STEP_TOKEN, stepToken)
            put(STEP_POSITION, stepPosition)
            putAll(stepCrossInputs)
            SELF_CACHE_NAMES.forEach { name -> put(name, inputSelf.getValue(name)) }
        }
        val outputs = HashMap<String, TensorBuffer>(10).apply {
            put(STEP_HIDDEN, stepHidden)
            stepLogits?.let { put(STEP_LOGITS, it) }
            SELF_CACHE_NAMES.forEach { name -> put(name, outputSelf.getValue(name)) }
        }
        stepModel.run(inputs, outputs, STEP_SIGNATURE)
        selfCacheFlip = !selfCacheFlip
        return stepHidden.readFloat().also {
            require(it.size == WhisperSplitDecoderTensors.ENCODER_WIDTH)
        }
    }

    override fun decodeTokens(tokens: List<Int>): String = tokenizer.decode(
        WhisperTimestampTokens.withoutTimestamps(
            tokens = tokens,
            timestampBeginTokenId = generationConfig.timestampBeginTokenId,
        ),
    )

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        synchronized(inferenceLock) {
            projection?.let { runCatching(it::close) }
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
        addAll(stepSelfCachesA.values)
        addAll(stepSelfCachesB.values)
        add(stepHidden)
        stepLogits?.let(::add)
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
        private const val STEP_LOGITS = "logits"
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

        /**
         * Opens the production KV pipeline from bundled APK assets. The asset
         * paths and hashes are pinned at build time via BuildConfig.
         */
        fun openProductionAssets(
            context: Context,
            encoderAssetPath: String,
            crossCacheAssetPath: String,
            stepAssetPath: String,
            projectionAssetPath: String,
            encoderPrecision: CompiledModel.GpuOptions.Precision,
            decoderPrecision: CompiledModel.GpuOptions.Precision,
            timestampsEnabled: Boolean = false,
        ): LiteRtKvCacheWhisperEngine = open(
            context = context,
            encoderModelPath = encoderAssetPath,
            crossCacheModelPath = crossCacheAssetPath,
            stepModelPath = stepAssetPath,
            projectionModelPath = projectionAssetPath,
            encoderPrecision = encoderPrecision,
            decoderPrecision = decoderPrecision,
            modelAssets = context.applicationContext.assets,
            timestampsEnabled = timestampsEnabled,
        )

        fun open(
            context: Context,
            encoderModelPath: String,
            crossCacheModelPath: String,
            stepModelPath: String,
            /**
             * Path of the separate FP32 CPU projection, or null when
             * [stepModelPath] is a fused model that emits `logits` directly.
             */
            projectionModelPath: String?,
            encoderPrecision: CompiledModel.GpuOptions.Precision,
            decoderPrecision: CompiledModel.GpuOptions.Precision,
            /** Non-null to resolve the model paths as APK assets. */
            modelAssets: android.content.res.AssetManager? = null,
            timestampsEnabled: Boolean = false,
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
            fun compile(path: String, precision: CompiledModel.GpuOptions.Precision) =
                if (modelAssets != null) {
                    CompiledModel.create(modelAssets, path, options(precision))
                } else {
                    CompiledModel.create(path, options(precision))
                }
            val encoder = compile(encoderModelPath, encoderPrecision)
            var cross: CompiledModel? = null
            var step: CompiledModel? = null
            var projection: ActiveRowProjection? = null
            val buffers = mutableListOf<TensorBuffer>()
            try {
                cross = compile(crossCacheModelPath, decoderPrecision)
                step = compile(stepModelPath, decoderPrecision)
                projection = projectionModelPath?.let { path ->
                    if (modelAssets != null) {
                        ActiveRowProjection.open(modelAssets, path, 3)
                    } else {
                        ActiveRowProjection.openFile(path, 3)
                    }
                }
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
                    stepLogits = if (projectionModelPath == null) {
                        output(step, STEP_LOGITS, STEP_SIGNATURE)
                    } else {
                        null
                    },
                    tokenizer = tokenizer,
                    generationConfig = config,
                    featureExtractor = WhisperFeatureExtractor(),
                    timestampsEnabled = timestampsEnabled,
                    encoderFeatures = input(encoder, ENCODE_INPUT, ENCODE_SIGNATURE),
                    encoderState = output(encoder, ENCODE_OUTPUT, ENCODE_SIGNATURE),
                    crossEncoderState = input(cross, CROSS_INPUT, CROSS_SIGNATURE),
                    crossOutputs = CROSS_CACHE_NAMES.associateWith { output(cross, it, CROSS_SIGNATURE) },
                    stepToken = input(step, STEP_TOKEN, STEP_SIGNATURE),
                    stepPosition = input(step, STEP_POSITION, STEP_SIGNATURE),
                    stepCrossInputs = CROSS_CACHE_NAMES.associateWith { input(step, it, STEP_SIGNATURE) },
                    // Ping-pong self-cache sets: A starts as the step input set
                    // and B as the output set; the roles alternate every step.
                    stepSelfCachesA = SELF_CACHE_NAMES.associateWith { input(step, it, STEP_SIGNATURE) },
                    stepSelfCachesB = SELF_CACHE_NAMES.associateWith { output(step, it, STEP_SIGNATURE) },
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
