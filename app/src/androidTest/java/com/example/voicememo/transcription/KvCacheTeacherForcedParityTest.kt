package com.example.voicememo.transcription

import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.CompiledModel
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/** Non-persisting two-phase teacher-forced parity probe for the KV-cache decoder. */
@RunWith(AndroidJUnit4::class)
class KvCacheTeacherForcedParityTest {
    @Test
    fun compareFailingChunk() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = safeName(requireNotNull(arguments.getString("runId")))
        val phase = requireNotNull(arguments.getString("phase"))
        require(phase == "full" || phase == "kv")
        val modelDirectory = File(context.filesDir, "gpu-models")
        val audioDirectory = File(context.filesDir, "gpu-audio")
        val encoder = requireFile(modelDirectory, arguments, "encoderModel")
        val audio = requireFile(audioDirectory, arguments, "audioFile")
        val language = WhisperLanguage.Forced(requireNotNull(arguments.getString("language")))
        val chunkIndex = requireNotNull(arguments.getString("chunkIndex")).toInt()
        val referenceTokens = requireNotNull(arguments.getString("referenceTokens"))
            .split(',')
            .filter(String::isNotBlank)
            .map(String::toInt)
        require(referenceTokens.isNotEmpty())

        val scratch = File(context.cacheDir, "kv-teacher-forced").apply { mkdirs() }
        val encoderStateFile = File(scratch, "$runId-encoder-state.bin")
        val fullLogitsFile = File(scratch, "$runId-full-logits.bin")
        val reportFile = File(scratch, "$runId-$phase.json")
        val samples = StreamingWavChunkSource.open(audio).use { source ->
            source.readChunk(chunkIndex).samples
        }
        val report = JSONObject()
            .put("run_id", runId)
            .put("phase", phase)
            .put("chunk_index", chunkIndex)
            .put("reference_token_count", referenceTokens.size)

        if (phase == "full") {
            val decoder = requireFile(modelDirectory, arguments, "decoderModel")
            LiteRtFullGpuWhisperEngine.open(
                context = context,
                allowSourceQuantizedFcConvOps = false,
                diagnosticEncoderModelFilePath = encoder.absolutePath,
                diagnosticDecoderModelFilePath = decoder.absolutePath,
                diagnosticEncoderPrecision = CompiledModel.GpuOptions.Precision.FP16,
                diagnosticDecoderPrecision = CompiledModel.GpuOptions.Precision.FP16,
            ).use { engine ->
                val encoderState = engine.encodeChunk(samples)
                writeFloats(encoderStateFile, encoderState)
                val rows = referenceTokens.indices.map { step ->
                    engine.teacherForcedLogitsRow(
                        encoderState = encoderState,
                        language = language,
                        referenceTokenPrefix = referenceTokens.take(step),
                    )
                }
                DataOutputStream(BufferedOutputStream(fullLogitsFile.outputStream())).use { output ->
                    output.writeInt(rows.size)
                    output.writeInt(rows.first().logits.size)
                    rows.forEach { row -> row.logits.forEach(output::writeFloat) }
                }
                report
                    .put("rows", summarizeRows(rows, referenceTokens))
                    .put("pss_kib", processPssKib())
            }
        } else {
            val cross = requireFile(modelDirectory, arguments, "crossModel")
            val step = requireFile(modelDirectory, arguments, "stepModel")
            // Absent when the fused step model emits logits directly.
            val projection = arguments.getString("projectionModel")?.let {
                requireFile(modelDirectory, arguments, "projectionModel")
            }
            val encoderState = readFloats(encoderStateFile)
            val fullRows = readRows(fullLogitsFile)
            require(fullRows.size == referenceTokens.size)
            LiteRtKvCacheWhisperEngine.open(
                context = context,
                encoderModelPath = encoder.absolutePath,
                crossCacheModelPath = cross.absolutePath,
                stepModelPath = step.absolutePath,
                projectionModelPath = projection?.absolutePath,
                encoderPrecision = CompiledModel.GpuOptions.Precision.FP16,
                decoderPrecision = CompiledModel.GpuOptions.Precision.FP16,
            ).use { engine ->
                val cachedRows = engine.teacherForcedLogitsRows(
                    preparedEncoderState = encoderState,
                    language = language,
                    referenceTokens = referenceTokens,
                )
                val comparisons = JSONArray()
                var firstTop1Mismatch: Int? = null
                var firstStableTop1Mismatch: Int? = null
                fullRows.indices.forEach { index ->
                    val full = WhisperTeacherForcedTokenRanking.forCpu(
                        logits = fullRows[index],
                        generationConfig = loadGenerationConfig(context.assets),
                        firstGeneratedToken = index == 0,
                    )
                    val cached = cachedRows[index]
                    val comparison = compareRows(index, referenceTokens[index], full, cached)
                    if (!comparison.getBoolean("top1_match") && firstTop1Mismatch == null) {
                        firstTop1Mismatch = index
                    }
                    if (
                        !comparison.getBoolean("stable_top1_match") &&
                        firstStableTop1Mismatch == null
                    ) {
                        firstStableTop1Mismatch = index
                    }
                    comparisons.put(comparison)
                }
                report
                    .put("first_top1_mismatch", firstTop1Mismatch ?: JSONObject.NULL)
                    .put("all_top1_match", firstTop1Mismatch == null)
                    .put(
                        "first_stable_top1_mismatch",
                        firstStableTop1Mismatch ?: JSONObject.NULL,
                    )
                    .put("all_stable_top1_match", firstStableTop1Mismatch == null)
                    .put("steps", comparisons)
                    .put("pss_kib", processPssKib())
            }
        }
        reportFile.writeText(report.toString(2))
    }

    private fun compareRows(
        index: Int,
        referenceToken: Int,
        full: WhisperTeacherForcedLogitsRow,
        cached: WhisperTeacherForcedLogitsRow,
    ): JSONObject {
        require(full.logits.size == cached.logits.size)
        val stableCachedTop1 = LiteRtKvCacheWhisperEngine.stableCachedTop1(cached)
        var maximumAbsoluteError = 0.0
        var squaredError = 0.0
        full.logits.indices.forEach { token ->
            val difference = (full.logits[token] - cached.logits[token]).toDouble()
            maximumAbsoluteError = maxOf(maximumAbsoluteError, kotlin.math.abs(difference))
            squaredError += difference * difference
        }
        return JSONObject()
            .put("step", index)
            .put("decoder_position", index + 4)
            .put("reference_token", referenceToken)
            .put("full_top1", full.selectableTop1?.tokenId ?: JSONObject.NULL)
            .put("cached_top1", cached.selectableTop1?.tokenId ?: JSONObject.NULL)
            .put("top1_match", full.selectableTop1?.tokenId == cached.selectableTop1?.tokenId)
            .put("cached_stable_top1", stableCachedTop1?.tokenId ?: JSONObject.NULL)
            .put("stable_top1_match", full.selectableTop1?.tokenId == stableCachedTop1?.tokenId)
            .put("full_top2", full.selectableTop2?.tokenId ?: JSONObject.NULL)
            .put("cached_top2", cached.selectableTop2?.tokenId ?: JSONObject.NULL)
            .put("full_margin", finite(full.selectableMargin))
            .put("cached_margin", finite(cached.selectableMargin))
            .put("full_reference_logit", finite(full.logits[referenceToken]))
            .put("cached_reference_logit", finite(cached.logits[referenceToken]))
            .put("maximum_absolute_logit_error", maximumAbsoluteError)
            .put("logit_rmse", sqrt(squaredError / full.logits.size))
    }

    private fun summarizeRows(
        rows: List<WhisperTeacherForcedLogitsRow>,
        referenceTokens: List<Int>,
    ): JSONArray = JSONArray().apply {
        rows.forEachIndexed { index, row ->
            put(
                JSONObject()
                    .put("step", index)
                    .put("decoder_position", index + 4)
                    .put("reference_token", referenceTokens[index])
                    .put("top1", row.selectableTop1?.tokenId ?: JSONObject.NULL)
                    .put("top2", row.selectableTop2?.tokenId ?: JSONObject.NULL)
                    .put("margin", finite(row.selectableMargin)),
            )
        }
    }

    private fun requireFile(directory: File, arguments: android.os.Bundle, key: String): File {
        val name = safeName(requireNotNull(arguments.getString(key)))
        return File(directory, name).also { require(it.isFile) { "Missing $key: $name" } }
    }

    private fun safeName(value: String): String {
        require(SAFE_NAME.matches(value)) { "Unsafe diagnostic name: $value" }
        return value
    }

    private fun loadGenerationConfig(assets: android.content.res.AssetManager) =
        assets.open(WhisperAssets.GENERATION_CONFIG).use(WhisperGenerationConfig::load)

    private fun writeFloats(file: File, values: FloatArray) {
        DataOutputStream(BufferedOutputStream(file.outputStream())).use { output ->
            output.writeInt(values.size)
            values.forEach(output::writeFloat)
        }
    }

    private fun readFloats(file: File): FloatArray =
        DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
            FloatArray(input.readInt()) { input.readFloat() }
        }

    private fun readRows(file: File): List<FloatArray> =
        DataInputStream(BufferedInputStream(file.inputStream())).use { input ->
            val rowCount = input.readInt()
            val width = input.readInt()
            List(rowCount) { FloatArray(width) { input.readFloat() } }
        }

    private fun processPssKib(): Int = Debug.MemoryInfo().also(Debug::getMemoryInfo).totalPss

    private fun finite(value: Float?): Any =
        if (value != null && value.isFinite()) value.toDouble() else JSONObject.NULL

    companion object {
        private val SAFE_NAME = Regex("[A-Za-z0-9._-]+")
    }
}
