package com.example.voicememo.transcription

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Encoder-only, non-persisting numerical parity diagnostic.
 *
 * Every runtime receives the exact same app-generated [128, 3000] log-mel
 * tensor. Decoder execution is deliberately absent. Final encoder tensors are
 * saved so host-side analysis can go deeper without rerunning the device.
 */
@RunWith(AndroidJUnit4::class)
class EncoderRuntimeParityDiagnosticTest {
    @Test
    fun reproduceFp16LayerNormVarianceOverflow() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val modelName = requireSafeName(arguments.getString(ARG_LAYER_NORM_MODEL_NAME), ".tflite")
        val modelSha = requireSha(arguments.getString(ARG_LAYER_NORM_MODEL_SHA256))
        val inputName = requireSafeName(arguments.getString(ARG_REPRO_INPUT_NAME), ".f32le")
        val inputSha = requireSha(arguments.getString(ARG_REPRO_INPUT_SHA256))
        val model = stagedFile(context, MODEL_DIRECTORY, modelName)
        val inputFile = stagedFile(context, STATE_DIRECTORY, inputName)
        require(sha256(model) == modelSha)
        require(sha256(inputFile) == inputSha)
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId-fp16-layernorm-repro.json")
        val temporaryReport = File(reportDirectory, "$runId-fp16-layernorm-repro.json.tmp")
        reportFile.delete()
        temporaryReport.delete()
        val report = JSONObject()
            .put("run_id", runId)
            .put("model_name", modelName)
            .put("model_sha256", modelSha)
            .put("input_name", inputName)
            .put("input_sha256", inputSha)
            .put("input_shape", JSONArray(listOf(1, REPRO_FRAMES, ENCODER_WIDTH)))
            .put("original_frame_indices", JSONArray(REPRO_ORIGINAL_FRAMES.toList()))
            .put("decoder_executed", false)
        try {
            val input = readFloatArray(inputFile)
            require(input.size == REPRO_FRAMES * ENCODER_WIDTH)
            val cpu = runLiteRtDiagnostic(
                name = LITERT_CPU_FP32,
                model = model,
                inputName = REPRO_INPUT,
                input = input,
                outputNames = FINAL_LAYER_NORM_OUTPUTS,
                accelerator = Accelerator.CPU,
                precision = null,
            )
            val gpu = runLiteRtDiagnostic(
                name = LITERT_GPU_FP16,
                model = model,
                inputName = REPRO_INPUT,
                input = input,
                outputNames = FINAL_LAYER_NORM_OUTPUTS,
                accelerator = Accelerator.GPU,
                precision = CompiledModel.GpuOptions.Precision.FP16,
            )
            val cpuMean = requireNotNull(cpu.outputs["mean"])
            val cpuVariance = requireNotNull(cpu.outputs["variance"])
            val cpuRsqrt = requireNotNull(cpu.outputs["rsqrt"])
            val gpuMean = requireNotNull(gpu.outputs["mean"])
            val gpuVariance = requireNotNull(gpu.outputs["variance"])
            val gpuRsqrt = requireNotNull(gpu.outputs["rsqrt"])
            val frames = JSONArray()
            REPRO_ORIGINAL_FRAMES.indices.forEach { index ->
                frames.put(
                    JSONObject()
                        .put("input_row", index)
                        .put("original_encoder_frame", REPRO_ORIGINAL_FRAMES[index])
                        .put("cpu_mean", jsonFloat(cpuMean[index]))
                        .put("gpu_fp16_mean", jsonFloat(gpuMean[index]))
                        .put("cpu_variance", jsonFloat(cpuVariance[index]))
                        .put("gpu_fp16_variance", jsonFloat(gpuVariance[index]))
                        .put("cpu_rsqrt", jsonFloat(cpuRsqrt[index]))
                        .put("gpu_fp16_rsqrt", jsonFloat(gpuRsqrt[index])),
                )
            }
            report
                .put("cpu_inference_millis", cpu.inferenceMillis)
                .put("gpu_fp16_inference_millis", gpu.inferenceMillis)
                .put("gpu_nonfinite_variance_count", gpuVariance.count { !it.isFinite() })
                .put("cpu_nonfinite_variance_count", cpuVariance.count { !it.isFinite() })
                .put("variance_comparison", compareFlatStates(cpuVariance, gpuVariance))
                .put("rsqrt_comparison", compareFlatStates(cpuRsqrt, gpuRsqrt))
                .put(
                    "output_comparison",
                    compareFlatStates(
                        requireNotNull(cpu.outputs["output"]),
                        requireNotNull(gpu.outputs["output"]),
                    ),
                )
                .put("frames", frames)
                .put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
        }
        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) { "Could not publish FP16 LayerNorm repro." }
    }

    @Test
    fun compareEncoderStage() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val runtimeVariant = arguments.getString(ARG_RUNTIME_VARIANT) ?: "standard"
        val modelName = requireSafeName(arguments.getString(ARG_ENCODER_MODEL_NAME), ".tflite")
        val expectedModelSha = requireSha(arguments.getString(ARG_ENCODER_MODEL_SHA256))
        val audioName = requireSafeName(arguments.getString(ARG_AUDIO_NAME), ".wav")
        val candidateNames = (arguments.getString(ARG_CANDIDATES) ?: DEFAULT_LITERT_CANDIDATES)
            .split(',')
            .map(String::trim)
            .filter(String::isNotEmpty)
        require(candidateNames.isNotEmpty() && candidateNames.all(LITERT_CANDIDATES::contains))
        val outputNames = (arguments.getString(ARG_STAGE_OUTPUTS) ?: DIAGNOSTIC_STAGE_OUTPUT)
            .split(',')
            .map(String::trim)
            .filter(String::isNotEmpty)
        require(outputNames.isNotEmpty() && outputNames.all(SAFE_STAGE_OUTPUT::matches))
        val model = stagedFile(context, MODEL_DIRECTORY, modelName)
        require(sha256(model) == expectedModelSha)
        val audioFile = stagedFile(context, AUDIO_DIRECTORY, audioName)
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId-stage.json")
        val temporaryReport = File(reportDirectory, "$runId-stage.json.tmp")
        reportFile.delete()
        temporaryReport.delete()
        val report = JSONObject()
            .put("run_id", runId)
            .put("runtime_variant", runtimeVariant)
            .put("encoder_model_name", modelName)
            .put("encoder_model_sha256", expectedModelSha)
            .put("audio_name", audioName)
            .put("audio_sha256", sha256(audioFile))
            .put("decoder_executed", false)
        try {
            val audio = StreamingWavChunkSource.open(audioFile).use { it.readChunk(0).samples }
            val mel = WhisperFeatureExtractor().extract(audio)
            report.put("mel_sha256", floatDigest(mel))
            val results = linkedMapOf<String, DiagnosticTensorResult>()
            candidateNames.forEach { name ->
                val (accelerator, precision) = liteRtRuntime(name)
                results[name] = runLiteRtDiagnostic(
                    name = name,
                    model = model,
                    inputName = DIAGNOSTIC_MEL_INPUT,
                    input = mel,
                    outputNames = outputNames,
                    accelerator = accelerator,
                    precision = precision,
                )
            }
            val reference = results[LITERT_CPU_FP32] ?: results.values.first()
            report
                .put("reference", reference.name)
                .put(
                    "candidates",
                    JSONArray().also { candidates ->
                        results.values.forEach { result ->
                            val outputs = JSONArray()
                            outputNames.forEach { outputName ->
                                val values = requireNotNull(result.outputs[outputName])
                                val stateFile = "$runId-${safeFilePart(result.name)}-$outputName.f32le"
                                writeFloatArray(File(reportDirectory, stateFile), values)
                                outputs.put(
                                    JSONObject()
                                        .put("name", outputName)
                                        .put("element_count", values.size)
                                        .put("state_file", stateFile)
                                        .put("state_sha256", floatDigest(values))
                                        .put(
                                            "vs_reference",
                                            compareFlatStates(
                                                requireNotNull(reference.outputs[outputName]),
                                                values,
                                            ),
                                        ),
                                )
                            }
                            candidates.put(
                                JSONObject()
                                    .put("name", result.name)
                                    .put("inference_millis", result.inferenceMillis)
                                    .put("outputs", outputs),
                            )
                        }
                    },
                )
                .put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
        }
        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) { "Could not publish encoder stage report." }
    }

    @Test
    fun compareFinalLayerNormCrossFeed() {
        val arguments = InstrumentationRegistry.getArguments()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val runtimeVariant = arguments.getString(ARG_RUNTIME_VARIANT) ?: "standard"
        val layerNormName = requireSafeName(arguments.getString(ARG_LAYER_NORM_MODEL_NAME), ".tflite")
        val layerNormSha = requireSha(arguments.getString(ARG_LAYER_NORM_MODEL_SHA256))
        val layerNormModel = stagedFile(context, MODEL_DIRECTORY, layerNormName)
        require(sha256(layerNormModel) == layerNormSha)
        fun state(argumentName: String, shaArgumentName: String): Pair<File, String> {
            val name = requireSafeName(arguments.getString(argumentName), ".f32le")
            val expectedSha = requireSha(arguments.getString(shaArgumentName))
            return stagedFile(context, STATE_DIRECTORY, name).also {
                require(sha256(it) == expectedSha) { "State file $name has the wrong SHA-256." }
            } to expectedSha
        }
        val (cpuBlock31File, cpuBlock31Sha) = state(ARG_CPU_BLOCK31_NAME, ARG_CPU_BLOCK31_SHA256)
        val (gpuBlock31File, gpuBlock31Sha) = state(ARG_GPU_BLOCK31_NAME, ARG_GPU_BLOCK31_SHA256)
        val (cpuFinalFile, cpuFinalSha) = state(ARG_CPU_FINAL_NAME, ARG_CPU_FINAL_SHA256)
        val (gpuFinalFile, gpuFinalSha) = state(ARG_GPU_FINAL_NAME, ARG_GPU_FINAL_SHA256)
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId-layernorm.json")
        val temporaryReport = File(reportDirectory, "$runId-layernorm.json.tmp")
        reportFile.delete()
        temporaryReport.delete()
        val report = JSONObject()
            .put("run_id", runId)
            .put("runtime_variant", runtimeVariant)
            .put("layer_norm_model_name", layerNormName)
            .put("layer_norm_model_sha256", layerNormSha)
            .put("source_cpu_block31_sha256", cpuBlock31Sha)
            .put("source_gpu_block31_sha256", gpuBlock31Sha)
            .put("source_cpu_final_sha256", cpuFinalSha)
            .put("source_gpu_final_sha256", gpuFinalSha)
            .put("decoder_executed", false)
        try {
            val cpuBlock31 = readFloatArray(cpuBlock31File)
            val gpuBlock31 = readFloatArray(gpuBlock31File)
            val cpuFinal = readFloatArray(cpuFinalFile)
            val gpuFinal = readFloatArray(gpuFinalFile)
            require(cpuBlock31.size == ENCODER_FLOATS && gpuBlock31.size == ENCODER_FLOATS)
            require(cpuFinal.size == ENCODER_FLOATS && gpuFinal.size == ENCODER_FLOATS)
            val inputs = mapOf(
                "cpu_block31" to cpuBlock31,
                "gpu_block31" to gpuBlock31,
            )
            val cases = linkedMapOf<String, DiagnosticTensorResult>()
            inputs.forEach { (inputName, state) ->
                cases["${inputName}_cpu_layernorm"] = runLiteRtDiagnostic(
                    name = "${inputName}_cpu_layernorm",
                    model = layerNormModel,
                    inputName = DIAGNOSTIC_LAYER_NORM_INPUT,
                    input = state,
                    outputNames = FINAL_LAYER_NORM_OUTPUTS,
                    accelerator = Accelerator.CPU,
                    precision = null,
                )
                cases["${inputName}_gpu_layernorm"] = runLiteRtDiagnostic(
                    name = "${inputName}_gpu_layernorm",
                    model = layerNormModel,
                    inputName = DIAGNOSTIC_LAYER_NORM_INPUT,
                    input = state,
                    outputNames = FINAL_LAYER_NORM_OUTPUTS,
                    accelerator = Accelerator.GPU,
                    precision = CompiledModel.GpuOptions.Precision.FP16,
                )
            }
            val reference = requireNotNull(cases["cpu_block31_cpu_layernorm"])
            val cpuInputGpuLayerNorm = requireNotNull(cases["cpu_block31_gpu_layernorm"])
            val gpuInputCpuLayerNorm = requireNotNull(cases["gpu_block31_cpu_layernorm"])
            val gpuInputGpuLayerNorm = requireNotNull(cases["gpu_block31_gpu_layernorm"])
            val caseRows = JSONArray()
            cases.values.forEach { result ->
                val outputs = JSONArray()
                FINAL_LAYER_NORM_OUTPUTS.forEach { outputName ->
                    val values = requireNotNull(result.outputs[outputName])
                    val stateFile = "$runId-${safeFilePart(result.name)}-$outputName.f32le"
                    writeFloatArray(File(reportDirectory, stateFile), values)
                    outputs.put(
                        JSONObject()
                            .put("name", outputName)
                            .put("element_count", values.size)
                            .put("state_file", stateFile)
                            .put("state_sha256", floatDigest(values))
                            .put(
                                "vs_cpu_input_cpu_layernorm",
                                compareFlatStates(requireNotNull(reference.outputs[outputName]), values),
                            ),
                    )
                }
                caseRows.put(
                    JSONObject()
                        .put("name", result.name)
                        .put("inference_millis", result.inferenceMillis)
                        .put("outputs", outputs),
                )
            }
            report
                .put(
                    "block31_gpu_vs_cpu",
                    compareStates(cpuBlock31, gpuBlock31),
                )
                .put(
                    "full_encoder_output_gpu_vs_cpu",
                    compareStates(cpuFinal, gpuFinal),
                )
                .put(
                    "standalone_cpu_reproduces_full_cpu",
                    compareFlatStates(
                        cpuFinal,
                        requireNotNull(reference.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                    ),
                )
                .put(
                    "standalone_gpu_reproduces_full_gpu",
                    compareFlatStates(
                        gpuFinal,
                        requireNotNull(gpuInputGpuLayerNorm.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                    ),
                )
                .put(
                    "gpu_layernorm_only_error",
                    compareFlatStates(
                        requireNotNull(reference.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                        requireNotNull(cpuInputGpuLayerNorm.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                    ),
                )
                .put(
                    "propagated_block31_error_after_cpu_layernorm",
                    compareFlatStates(
                        requireNotNull(reference.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                        requireNotNull(gpuInputCpuLayerNorm.outputs[DIAGNOSTIC_LAYER_NORM_OUTPUT]),
                    ),
                )
                .put("layer_norm_cases", caseRows)
                .put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
        }
        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) { "Could not publish final LayerNorm report." }
    }

    @Test
    fun compareEncoderBlockBoundaries() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val runtimeVariant = arguments.getString(ARG_RUNTIME_VARIANT) ?: "standard"
        val modelName = requireSafeName(arguments.getString(ARG_ENCODER_MODEL_NAME), ".tflite")
        val expectedModelSha = requireSha(arguments.getString(ARG_ENCODER_MODEL_SHA256))
        val audioName = requireSafeName(arguments.getString(ARG_AUDIO_NAME), ".wav")
        val whisperModelName = requireSafeName(
            arguments.getString(ARG_WHISPER_MODEL_NAME) ?: DEFAULT_WHISPER_MODEL,
            ".bin",
        )
        val candidateNames = (arguments.getString(ARG_CANDIDATES) ?: DEFAULT_BLOCK_CANDIDATES)
            .split(',')
            .map(String::trim)
            .filter(String::isNotEmpty)
        require(candidateNames.all(BLOCK_CANDIDATES::contains))
        val encoderModel = stagedFile(context, MODEL_DIRECTORY, modelName)
        require(sha256(encoderModel) == expectedModelSha)
        val audioFile = stagedFile(context, AUDIO_DIRECTORY, audioName)
        val whisperModel = stagedFile(context, WHISPER_MODEL_DIRECTORY, whisperModelName)
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId-blocks.json")
        val temporaryReport = File(reportDirectory, "$runId-blocks.json.tmp")
        reportFile.delete()
        temporaryReport.delete()
        val report = JSONObject()
            .put("run_id", runId)
            .put("runtime_variant", runtimeVariant)
            .put("encoder_model_name", modelName)
            .put("encoder_model_sha256", expectedModelSha)
            .put("audio_name", audioName)
            .put("audio_sha256", sha256(audioFile))
            .put("decoder_executed", false)
            .put("boundaries", JSONArray(BLOCK_OUTPUT_NAMES))
        try {
            val audio = StreamingWavChunkSource.open(audioFile).use { it.readChunk(0).samples }
            val mel = WhisperFeatureExtractor().extract(audio)
            val orderedCandidates = if (LITERT_CPU_FP32 in candidateNames) {
                listOf(LITERT_CPU_FP32) + candidateNames.filterNot { it == LITERT_CPU_FP32 }
            } else {
                candidateNames
            }
            val referenceName = orderedCandidates.first()
            val referenceFiles = mutableMapOf<String, File>()
            val candidates = JSONArray()
            for (name in orderedCandidates) {
                run {
                    val result = when (name) {
                    LITERT_CPU_FP32 -> runLiteRtBlocks(
                        name,
                        encoderModel,
                        mel,
                        Accelerator.CPU,
                        null,
                    )
                    LITERT_GPU_FP16 -> runLiteRtBlocks(
                        name,
                        encoderModel,
                        mel,
                        Accelerator.GPU,
                        CompiledModel.GpuOptions.Precision.FP16,
                    )
                    WHISPER_CPP_VULKAN -> runWhisperCppBlocks(
                        name,
                        whisperModel,
                        mel,
                    )
                    else -> error("Unhandled block candidate $name")
                }
                    val boundaries = JSONArray()
                    BLOCK_OUTPUT_NAMES.forEach { boundary ->
                    val state = requireNotNull(result.states[boundary])
                    val stateFile = "$runId-${safeFilePart(result.name)}-$boundary.f32le"
                    val outputFile = File(reportDirectory, stateFile)
                    writeFloatArray(outputFile, state)
                    val comparison = if (result.name == referenceName) {
                        referenceFiles[boundary] = outputFile
                        compareStates(state, state)
                    } else {
                        compareStateFile(requireNotNull(referenceFiles[boundary]), state)
                    }
                    boundaries.put(
                        JSONObject()
                            .put("name", boundary)
                            .put("state_file", stateFile)
                            .put("state_sha256", floatDigest(state))
                            .put(
                                "vs_reference",
                                comparison,
                            ),
                    )
                }
                    candidates.put(
                        JSONObject()
                            .put("name", result.name)
                            .put("encoder_millis", result.encoderMillis)
                            .put("boundaries", boundaries),
                    )
                }
                System.gc()
            }
            report
                .put("reference", referenceName)
                .put("candidates", candidates)
                .put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
        }
        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) { "Could not publish encoder block report." }
    }

    @Test
    fun compareFinalEncoderStates() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = requireSafeRunId(arguments.getString(ARG_RUN_ID))
        val runtimeVariant = arguments.getString(ARG_RUNTIME_VARIANT) ?: "standard"
        val modelName = requireSafeName(
            arguments.getString(ARG_ENCODER_MODEL_NAME),
            ".tflite",
        )
        val expectedModelSha = requireSha(arguments.getString(ARG_ENCODER_MODEL_SHA256))
        val audioName = requireSafeName(arguments.getString(ARG_AUDIO_NAME), ".wav")
        val whisperModelName = requireSafeName(
            arguments.getString(ARG_WHISPER_MODEL_NAME) ?: DEFAULT_WHISPER_MODEL,
            ".bin",
        )
        val candidateNames = (arguments.getString(ARG_CANDIDATES) ?: DEFAULT_CANDIDATES)
            .split(',')
            .map(String::trim)
            .filter(String::isNotEmpty)
        require(candidateNames.isNotEmpty() && candidateNames.distinct().size == candidateNames.size)
        require(candidateNames.all(SUPPORTED_CANDIDATES::contains)) {
            "Unsupported encoder candidate in ${candidateNames.joinToString()}."
        }

        val encoderModel = stagedFile(context, MODEL_DIRECTORY, modelName)
        require(sha256(encoderModel) == expectedModelSha) {
            "Staged encoder SHA-256 does not match."
        }
        val audioFile = stagedFile(context, AUDIO_DIRECTORY, audioName)
        val whisperModel = stagedFile(context, WHISPER_MODEL_DIRECTORY, whisperModelName)
        val reportDirectory = File(context.cacheDir, REPORT_DIRECTORY).apply { mkdirs() }
        val reportFile = File(reportDirectory, "$runId.json")
        val temporaryReport = File(reportDirectory, "$runId.json.tmp")
        reportFile.delete()
        temporaryReport.delete()

        val report = JSONObject()
            .put("run_id", runId)
            .put("runtime_variant", runtimeVariant)
            .put("encoder_model_name", modelName)
            .put("encoder_model_sha256", expectedModelSha)
            .put("audio_name", audioName)
            .put("audio_sha256", sha256(audioFile))
            .put("whisper_model_name", whisperModelName)
            .put("whisper_model_sha256", sha256(whisperModel))
            .put("mel_shape", JSONArray(listOf(1, MEL_BINS, MEL_FRAMES)))
            .put("encoder_shape", JSONArray(listOf(1, ENCODER_FRAMES, ENCODER_WIDTH)))
            .put("decoder_executed", false)

        try {
            val audio = StreamingWavChunkSource.open(audioFile).use { source ->
                source.readChunk(0).samples
            }
            val featureStarted = System.nanoTime()
            val mel = WhisperFeatureExtractor().extract(audio)
            report
                .put("audio_samples", audio.size)
                .put("feature_extraction_millis", elapsedMillis(featureStarted))
                .put("mel_sha256", floatDigest(mel))

            val results = linkedMapOf<String, EncoderResult>()
            candidateNames.forEach { name ->
                val result = when (name) {
                    LITERT_CPU_FP32 -> runLiteRt(
                        name = name,
                        model = encoderModel,
                        mel = mel,
                        accelerator = Accelerator.CPU,
                        precision = null,
                    )
                    LITERT_GPU_FP32 -> runLiteRt(
                        name = name,
                        model = encoderModel,
                        mel = mel,
                        accelerator = Accelerator.GPU,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                    )
                    LITERT_GPU_FP16 -> runLiteRt(
                        name = name,
                        model = encoderModel,
                        mel = mel,
                        accelerator = Accelerator.GPU,
                        precision = CompiledModel.GpuOptions.Precision.FP16,
                    )
                    WHISPER_CPP_CPU -> runWhisperCpp(
                        name,
                        whisperModel,
                        mel,
                        WhisperCppBenchmarkBackend.CPU,
                    )
                    WHISPER_CPP_VULKAN -> runWhisperCpp(
                        name,
                        whisperModel,
                        mel,
                        WhisperCppBenchmarkBackend.VULKAN,
                    )
                    else -> error("Unhandled encoder candidate $name")
                }
                require(result.state.size == ENCODER_FLOATS) {
                    "$name returned ${result.state.size} encoder values; expected $ENCODER_FLOATS."
                }
                val stateFile = File(reportDirectory, "$runId-${safeFilePart(name)}.f32le")
                writeFloatArray(stateFile, result.state)
                results[name] = result.copy(stateFile = stateFile.name)
            }

            val reference = results[LITERT_CPU_FP32] ?: results.values.first()
            val candidates = JSONArray()
            results.values.forEach { result ->
                candidates.put(
                    JSONObject()
                        .put("name", result.name)
                        .put("encoder_millis", result.encoderMillis)
                        .put("state_sha256", floatDigest(result.state))
                        .put("state_file", result.stateFile)
                        .put(
                            "vs_reference",
                            compareStates(reference.state, result.state),
                        ),
                )
            }
            val pairwise = JSONArray()
            addPairwise(pairwise, results, LITERT_GPU_FP32, LITERT_GPU_FP16)
            addPairwise(pairwise, results, WHISPER_CPP_CPU, WHISPER_CPP_VULKAN)
            addPairwise(pairwise, results, LITERT_GPU_FP32, WHISPER_CPP_VULKAN)
            report
                .put("reference", reference.name)
                .put("candidates", candidates)
                .put("pairwise", pairwise)
                .put("status", "completed")
        } catch (error: Throwable) {
            report
                .put("status", "failed")
                .put("error_type", error.javaClass.name)
                .put("error_message", error.message ?: error.toString())
        }

        temporaryReport.writeText(report.toString(2))
        check(temporaryReport.renameTo(reportFile)) { "Could not publish encoder report." }
    }

    private fun runLiteRt(
        name: String,
        model: File,
        mel: FloatArray,
        accelerator: Accelerator,
        precision: CompiledModel.GpuOptions.Precision?,
    ): EncoderResult {
        val options = CompiledModel.Options(accelerator).apply {
            if (precision != null) {
                gpuOptions = CompiledModel.GpuOptions(
                    constantTensorSharing = true,
                    allowSrcQuantizedFcConvOps = false,
                    precision = precision,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                )
            }
        }
        return CompiledModel.create(model.absolutePath, options).use { compiled ->
            val input = compiled.createInputBuffer(ENCODE_INPUT, ENCODE_SIGNATURE)
            val output = compiled.createOutputBuffer(ENCODE_OUTPUT, ENCODE_SIGNATURE)
            try {
                input.writeFloat(mel)
                val started = System.nanoTime()
                compiled.run(
                    mapOf(ENCODE_INPUT to input),
                    mapOf(ENCODE_OUTPUT to output),
                    ENCODE_SIGNATURE,
                )
                val state = output.readFloat()
                EncoderResult(name, elapsedMillis(started), state)
            } finally {
                output.close()
                input.close()
            }
        }
    }

    private fun runWhisperCpp(
        name: String,
        model: File,
        mel: FloatArray,
        backend: WhisperCppBenchmarkBackend,
    ): EncoderResult = WhisperCppBenchmarkEngine.open(
        modelFile = model,
        threadCount = 4,
        backend = backend,
    ).use { engine ->
        val state = engine.encodeMel(mel)
        EncoderResult(name, engine.encoderMillis(), state)
    }

    private fun runLiteRtBlocks(
        name: String,
        model: File,
        mel: FloatArray,
        accelerator: Accelerator,
        precision: CompiledModel.GpuOptions.Precision?,
        outputNames: List<String> = BLOCK_OUTPUT_NAMES,
    ): BlockEncoderResult {
        val options = CompiledModel.Options(accelerator).apply {
            if (precision != null) {
                gpuOptions = CompiledModel.GpuOptions(
                    constantTensorSharing = true,
                    allowSrcQuantizedFcConvOps = false,
                    precision = precision,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                )
            }
        }
        return CompiledModel.create(model.absolutePath, options).use { compiled ->
            val input = compiled.createInputBuffer(ENCODE_INPUT, ENCODE_SIGNATURE)
            val outputs = outputNames.associateWith { outputName ->
                compiled.createOutputBuffer(outputName, ENCODE_SIGNATURE)
            }
            try {
                input.writeFloat(mel)
                val started = System.nanoTime()
                compiled.run(
                    mapOf(ENCODE_INPUT to input),
                    outputs,
                    ENCODE_SIGNATURE,
                )
                val millis = elapsedMillis(started)
                BlockEncoderResult(
                    name,
                    millis,
                    outputs.mapValues { (_, buffer) -> buffer.readFloat() },
                )
            } finally {
                outputs.values.forEach(TensorBuffer::close)
                input.close()
            }
        }
    }

    private fun runLiteRtDiagnostic(
        name: String,
        model: File,
        inputName: String,
        input: FloatArray,
        outputNames: List<String>,
        accelerator: Accelerator,
        precision: CompiledModel.GpuOptions.Precision?,
    ): DiagnosticTensorResult {
        val options = CompiledModel.Options(accelerator).apply {
            if (precision != null) {
                gpuOptions = CompiledModel.GpuOptions(
                    constantTensorSharing = true,
                    allowSrcQuantizedFcConvOps = false,
                    precision = precision,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                )
            }
        }
        return CompiledModel.create(model.absolutePath, options).use { compiled ->
            val inputBuffer = compiled.createInputBuffer(inputName, DIAGNOSTIC_SIGNATURE)
            val outputBuffers = outputNames.associateWith { outputName ->
                compiled.createOutputBuffer(outputName, DIAGNOSTIC_SIGNATURE)
            }
            try {
                inputBuffer.writeFloat(input)
                val started = System.nanoTime()
                compiled.run(
                    mapOf(inputName to inputBuffer),
                    outputBuffers,
                    DIAGNOSTIC_SIGNATURE,
                )
                DiagnosticTensorResult(
                    name,
                    elapsedMillis(started),
                    outputBuffers.mapValues { (_, buffer) -> buffer.readFloat() },
                )
            } finally {
                outputBuffers.values.forEach(TensorBuffer::close)
                inputBuffer.close()
            }
        }
    }

    private fun liteRtRuntime(
        name: String,
    ): Pair<Accelerator, CompiledModel.GpuOptions.Precision?> = when (name) {
        LITERT_CPU_FP32 -> Accelerator.CPU to null
        LITERT_GPU_FP32 -> Accelerator.GPU to CompiledModel.GpuOptions.Precision.FP32
        LITERT_GPU_FP16 -> Accelerator.GPU to CompiledModel.GpuOptions.Precision.FP16
        else -> error("Unsupported LiteRT diagnostic runtime $name")
    }

    private fun runWhisperCppBlocks(
        name: String,
        model: File,
        mel: FloatArray,
    ): BlockEncoderResult = WhisperCppBenchmarkEngine.open(
        modelFile = model,
        threadCount = 4,
        backend = WhisperCppBenchmarkBackend.VULKAN,
    ).use { engine ->
        val finalState = engine.encodeMel(mel)
        val blocks = engine.encoderBlockStates()
        require(blocks.size == BLOCK_OUTPUT_NAMES.size - 1) {
            "whisper.cpp returned ${blocks.size} encoder boundaries."
        }
        BlockEncoderResult(
            name,
            engine.encoderMillis(),
            buildMap {
                BLOCK_OUTPUT_NAMES.dropLast(1).forEachIndexed { index, boundary ->
                    put(boundary, blocks[index])
                }
                put(ENCODE_OUTPUT, finalState)
            },
        )
    }

    private fun addPairwise(
        output: JSONArray,
        results: Map<String, EncoderResult>,
        leftName: String,
        rightName: String,
    ) {
        val left = results[leftName] ?: return
        val right = results[rightName] ?: return
        output.put(
            JSONObject()
                .put("left", leftName)
                .put("right", rightName)
                .put("comparison", compareStates(left.state, right.state)),
        )
    }

    private fun compareStates(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size)
        var exact = 0
        var finite = 0
        var maxAbs = 0.0
        var maxIndex = -1
        var sumAbs = 0.0
        var sumSquared = 0.0
        var dot = 0.0
        var referenceSquared = 0.0
        var candidateSquared = 0.0
        val frameSquared = DoubleArray(ENCODER_FRAMES)
        val channelSquared = DoubleArray(ENCODER_WIDTH)
        for (index in reference.indices) {
            val expected = reference[index]
            val actual = candidate[index]
            if (expected.toRawBits() == actual.toRawBits()) exact++
            if (!expected.isFinite() || !actual.isFinite()) continue
            finite++
            val difference = actual.toDouble() - expected.toDouble()
            val absolute = abs(difference)
            if (absolute > maxAbs) {
                maxAbs = absolute
                maxIndex = index
            }
            sumAbs += absolute
            sumSquared += difference * difference
            dot += expected.toDouble() * actual.toDouble()
            referenceSquared += expected.toDouble() * expected.toDouble()
            candidateSquared += actual.toDouble() * actual.toDouble()
            val frame = index / ENCODER_WIDTH
            val channel = index % ENCODER_WIDTH
            frameSquared[frame] += difference * difference
            channelSquared[channel] += difference * difference
        }
        val worstFrames = frameSquared.indices
            .sortedByDescending(frameSquared::get)
            .take(20)
        val worstChannels = channelSquared.indices
            .sortedByDescending(channelSquared::get)
            .take(20)
        return JSONObject()
            .put("element_count", reference.size)
            .put("finite_pair_count", finite)
            .put("exact_element_count", exact)
            .put("exact_fraction", exact.toDouble() / reference.size)
            .put("max_absolute_error", maxAbs)
            .put("max_error_index", maxIndex)
            .put("max_error_frame", if (maxIndex >= 0) maxIndex / ENCODER_WIDTH else -1)
            .put("max_error_channel", if (maxIndex >= 0) maxIndex % ENCODER_WIDTH else -1)
            .put("mean_absolute_error", if (finite > 0) sumAbs / finite else JSONObject.NULL)
            .put("rmse", if (finite > 0) sqrt(sumSquared / finite) else JSONObject.NULL)
            .put(
                "cosine_similarity",
                if (referenceSquared > 0.0 && candidateSquared > 0.0) {
                    dot / sqrt(referenceSquared * candidateSquared)
                } else {
                    JSONObject.NULL
                },
            )
            .put(
                "worst_frames_by_rmse",
                JSONArray().also { rows ->
                    worstFrames.forEach { frame ->
                        rows.put(
                            JSONObject()
                                .put("frame", frame)
                                .put("rmse", sqrt(frameSquared[frame] / ENCODER_WIDTH)),
                        )
                    }
                },
            )
            .put(
                "worst_channels_by_rmse",
                JSONArray().also { rows ->
                    worstChannels.forEach { channel ->
                        rows.put(
                            JSONObject()
                                .put("channel", channel)
                                .put("rmse", sqrt(channelSquared[channel] / ENCODER_FRAMES)),
                        )
                    }
                },
            )
    }

    private fun compareFlatStates(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size)
        var exact = 0
        var finite = 0
        var maxAbs = 0.0
        var maxIndex = -1
        var sumAbs = 0.0
        var sumSquared = 0.0
        var dot = 0.0
        var referenceSquared = 0.0
        var candidateSquared = 0.0
        reference.indices.forEach { index ->
            val expected = reference[index]
            val actual = candidate[index]
            if (expected.toRawBits() == actual.toRawBits()) exact++
            if (!expected.isFinite() || !actual.isFinite()) return@forEach
            finite++
            val difference = actual.toDouble() - expected.toDouble()
            val absolute = abs(difference)
            if (absolute > maxAbs) {
                maxAbs = absolute
                maxIndex = index
            }
            sumAbs += absolute
            sumSquared += difference * difference
            dot += expected.toDouble() * actual.toDouble()
            referenceSquared += expected.toDouble() * expected.toDouble()
            candidateSquared += actual.toDouble() * actual.toDouble()
        }
        return JSONObject()
            .put("element_count", reference.size)
            .put("finite_pair_count", finite)
            .put("exact_element_count", exact)
            .put("exact_fraction", exact.toDouble() / reference.size)
            .put("max_absolute_error", maxAbs)
            .put("max_error_index", maxIndex)
            .put("mean_absolute_error", if (finite > 0) sumAbs / finite else JSONObject.NULL)
            .put("rmse", if (finite > 0) sqrt(sumSquared / finite) else JSONObject.NULL)
            .put(
                "cosine_similarity",
                if (referenceSquared > 0.0 && candidateSquared > 0.0) {
                    dot / sqrt(referenceSquared * candidateSquared)
                } else {
                    JSONObject.NULL
                },
            )
    }

    private fun compareStateFile(referenceFile: File, candidate: FloatArray): JSONObject {
        require(referenceFile.length() == candidate.size.toLong() * Float.SIZE_BYTES)
        var exact = 0
        var finite = 0
        var maxAbs = 0.0
        var maxIndex = -1
        var sumAbs = 0.0
        var sumSquared = 0.0
        var dot = 0.0
        var referenceSquared = 0.0
        var candidateSquared = 0.0
        val frameSquared = DoubleArray(ENCODER_FRAMES)
        val channelSquared = DoubleArray(ENCODER_WIDTH)
        var index = 0
        FileInputStream(referenceFile).channel.use { channel ->
            val bytes = ByteBuffer.allocateDirect(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN)
            while (true) {
                val count = channel.read(bytes)
                if (count < 0) break
                bytes.flip()
                while (bytes.remaining() >= Float.SIZE_BYTES) {
                    val expected = bytes.float
                    val actual = candidate[index]
                    if (expected.toRawBits() == actual.toRawBits()) exact++
                    if (expected.isFinite() && actual.isFinite()) {
                        finite++
                        val difference = actual.toDouble() - expected.toDouble()
                        val absolute = abs(difference)
                        if (absolute > maxAbs) {
                            maxAbs = absolute
                            maxIndex = index
                        }
                        sumAbs += absolute
                        sumSquared += difference * difference
                        dot += expected.toDouble() * actual.toDouble()
                        referenceSquared += expected.toDouble() * expected.toDouble()
                        candidateSquared += actual.toDouble() * actual.toDouble()
                        frameSquared[index / ENCODER_WIDTH] += difference * difference
                        channelSquared[index % ENCODER_WIDTH] += difference * difference
                    }
                    index++
                }
                bytes.compact()
            }
            require(bytes.position() == 0) { "Reference state ends with a partial float." }
        }
        require(index == candidate.size)
        val worstFrames = frameSquared.indices.sortedByDescending(frameSquared::get).take(20)
        val worstChannels = channelSquared.indices.sortedByDescending(channelSquared::get).take(20)
        return JSONObject()
            .put("element_count", candidate.size)
            .put("finite_pair_count", finite)
            .put("exact_element_count", exact)
            .put("exact_fraction", exact.toDouble() / candidate.size)
            .put("max_absolute_error", maxAbs)
            .put("max_error_index", maxIndex)
            .put("max_error_frame", if (maxIndex >= 0) maxIndex / ENCODER_WIDTH else -1)
            .put("max_error_channel", if (maxIndex >= 0) maxIndex % ENCODER_WIDTH else -1)
            .put("mean_absolute_error", if (finite > 0) sumAbs / finite else JSONObject.NULL)
            .put("rmse", if (finite > 0) sqrt(sumSquared / finite) else JSONObject.NULL)
            .put(
                "cosine_similarity",
                if (referenceSquared > 0.0 && candidateSquared > 0.0) {
                    dot / sqrt(referenceSquared * candidateSquared)
                } else {
                    JSONObject.NULL
                },
            )
            .put(
                "worst_frames_by_rmse",
                JSONArray().also { rows ->
                    worstFrames.forEach { frame ->
                        rows.put(
                            JSONObject()
                                .put("frame", frame)
                                .put("rmse", sqrt(frameSquared[frame] / ENCODER_WIDTH)),
                        )
                    }
                },
            )
            .put(
                "worst_channels_by_rmse",
                JSONArray().also { rows ->
                    worstChannels.forEach { channel ->
                        rows.put(
                            JSONObject()
                                .put("channel", channel)
                                .put("rmse", sqrt(channelSquared[channel] / ENCODER_FRAMES)),
                        )
                    }
                },
            )
    }

    private fun writeFloatArray(file: File, values: FloatArray) {
        val bytes = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        bytes.asFloatBuffer().put(values)
        file.writeBytes(bytes.array())
    }

    private fun readFloatArray(file: File): FloatArray {
        require(file.length() % Float.SIZE_BYTES == 0L) { "State file contains a partial float." }
        require(file.length() <= Int.MAX_VALUE) { "State file is too large." }
        val bytes = file.readBytes()
        return FloatArray(bytes.size / Float.SIZE_BYTES).also { values ->
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
        }
    }

    private fun floatDigest(values: FloatArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { value ->
            bytes.clear()
            bytes.putFloat(value)
            digest.update(bytes.array())
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun stagedFile(context: Context, directory: String, name: String): File {
        val parent = File(context.filesDir, directory).canonicalFile
        return File(parent, name).canonicalFile.also { file ->
            require(file.parentFile == parent && file.isFile) { "Staged file $name is missing." }
        }
    }

    private fun requireSafeRunId(value: String?): String = requireNotNull(value).also {
        require(SAFE_RUN_ID.matches(it)) { "Invalid run ID." }
    }

    private fun requireSafeName(value: String?, suffix: String): String = requireNotNull(value).also {
        require(SAFE_NAME.matches(it) && it.endsWith(suffix)) { "Invalid staged file name." }
    }

    private fun requireSha(value: String?): String = requireNotNull(value).also {
        require(SAFE_SHA.matches(it)) { "Invalid SHA-256." }
    }

    private fun safeFilePart(value: String): String = value.replace(Regex("[^A-Za-z0-9._-]"), "-")

    private fun jsonFloat(value: Float): Any = when {
        value.isNaN() -> "NaN"
        value == Float.POSITIVE_INFINITY -> "+Infinity"
        value == Float.NEGATIVE_INFINITY -> "-Infinity"
        else -> value.toDouble()
    }

    private fun elapsedMillis(started: Long): Long = (System.nanoTime() - started) / 1_000_000L

    private data class EncoderResult(
        val name: String,
        val encoderMillis: Long,
        val state: FloatArray,
        val stateFile: String? = null,
    )

    private data class BlockEncoderResult(
        val name: String,
        val encoderMillis: Long,
        val states: Map<String, FloatArray>,
    )

    private data class DiagnosticTensorResult(
        val name: String,
        val inferenceMillis: Long,
        val outputs: Map<String, FloatArray>,
    )

    companion object {
        private const val ARG_RUN_ID = "runId"
        private const val ARG_RUNTIME_VARIANT = "runtimeVariant"
        private const val ARG_ENCODER_MODEL_NAME = "encoderModelName"
        private const val ARG_ENCODER_MODEL_SHA256 = "encoderModelSha256"
        private const val ARG_AUDIO_NAME = "audioName"
        private const val ARG_WHISPER_MODEL_NAME = "whisperModelName"
        private const val ARG_CANDIDATES = "candidates"
        private const val ARG_STAGE_OUTPUTS = "stageOutputs"
        private const val ARG_LAYER_NORM_MODEL_NAME = "layerNormModelName"
        private const val ARG_LAYER_NORM_MODEL_SHA256 = "layerNormModelSha256"
        private const val ARG_CPU_BLOCK31_NAME = "cpuBlock31Name"
        private const val ARG_CPU_BLOCK31_SHA256 = "cpuBlock31Sha256"
        private const val ARG_GPU_BLOCK31_NAME = "gpuBlock31Name"
        private const val ARG_GPU_BLOCK31_SHA256 = "gpuBlock31Sha256"
        private const val ARG_CPU_FINAL_NAME = "cpuFinalName"
        private const val ARG_CPU_FINAL_SHA256 = "cpuFinalSha256"
        private const val ARG_GPU_FINAL_NAME = "gpuFinalName"
        private const val ARG_GPU_FINAL_SHA256 = "gpuFinalSha256"
        private const val ARG_REPRO_INPUT_NAME = "reproInputName"
        private const val ARG_REPRO_INPUT_SHA256 = "reproInputSha256"
        private const val MODEL_DIRECTORY = "gpu-models"
        private const val AUDIO_DIRECTORY = "gpu-audio"
        private const val WHISPER_MODEL_DIRECTORY = "whispercpp-models"
        private const val STATE_DIRECTORY = "encoder-states"
        private const val REPORT_DIRECTORY = "encoder-parity"
        private const val ENCODE_SIGNATURE = "encode"
        private const val ENCODE_INPUT = "args_0"
        private const val ENCODE_OUTPUT = "output_0"
        private const val BLOCK_31_OUTPUT = "block_31"
        private const val DIAGNOSTIC_SIGNATURE = "diagnostic"
        private const val DIAGNOSTIC_MEL_INPUT = "mel"
        private const val DIAGNOSTIC_STAGE_OUTPUT = "stage"
        private const val DIAGNOSTIC_LAYER_NORM_INPUT = "block_31"
        private const val DIAGNOSTIC_LAYER_NORM_OUTPUT = "output"
        private const val REPRO_INPUT = "input"
        private const val REPRO_FRAMES = 6
        private const val MEL_BINS = 128
        private const val MEL_FRAMES = 3_000
        private const val ENCODER_FRAMES = 1_500
        private const val ENCODER_WIDTH = 1_280
        private const val ENCODER_FLOATS = ENCODER_FRAMES * ENCODER_WIDTH
        private const val LITERT_CPU_FP32 = "litert-cpu-fp32"
        private const val LITERT_GPU_FP32 = "litert-opencl-fp32"
        private const val LITERT_GPU_FP16 = "litert-opencl-fp16"
        private const val WHISPER_CPP_CPU = "whispercpp-cpu-fp16-weights"
        private const val WHISPER_CPP_VULKAN = "whispercpp-vulkan-fp16-weights"
        private const val DEFAULT_WHISPER_MODEL = "ggml-large-v3-turbo.bin"
        private const val DEFAULT_CANDIDATES =
            "$LITERT_CPU_FP32,$LITERT_GPU_FP32,$LITERT_GPU_FP16,$WHISPER_CPP_CPU,$WHISPER_CPP_VULKAN"
        private const val DEFAULT_BLOCK_CANDIDATES =
            "$LITERT_CPU_FP32,$LITERT_GPU_FP16,$WHISPER_CPP_VULKAN"
        private const val DEFAULT_LITERT_CANDIDATES =
            "$LITERT_CPU_FP32,$LITERT_GPU_FP32,$LITERT_GPU_FP16"
        private val FINAL_LAYER_NORM_OUTPUTS = listOf("mean", "variance", "rsqrt", "output")
        private val REPRO_ORIGINAL_FRAMES = intArrayOf(66, 1379, 1385, 1391, 1455, 1456)
        private val BLOCK_OUTPUT_NAMES = listOf(
            "conv_stem",
            "block_00",
            "block_03",
            "block_07",
            "block_15",
            "block_23",
            "block_24",
            "block_25",
            "block_26",
            "block_27",
            "block_28",
            "block_29",
            "block_30",
            "block_31",
            ENCODE_OUTPUT,
        )
        private val BLOCK_CANDIDATES = setOf(
            LITERT_CPU_FP32,
            LITERT_GPU_FP16,
            WHISPER_CPP_VULKAN,
        )
        private val LITERT_CANDIDATES = setOf(
            LITERT_CPU_FP32,
            LITERT_GPU_FP32,
            LITERT_GPU_FP16,
        )
        private val SUPPORTED_CANDIDATES = setOf(
            LITERT_CPU_FP32,
            LITERT_GPU_FP32,
            LITERT_GPU_FP16,
            WHISPER_CPP_CPU,
            WHISPER_CPP_VULKAN,
        )
        private val SAFE_RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
        private val SAFE_NAME = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        private val SAFE_SHA = Regex("[0-9a-f]{64}")
        private val SAFE_STAGE_OUTPUT = Regex("[A-Za-z0-9_]+")
    }
}
