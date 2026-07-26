package com.example.voicememo.transcription

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtSplitDecoderDiagnosticTest {
    private val config = WhisperGenerationConfig(
        eosTokenId = 6,
        decoderStartTokenId = 7,
        transcribeTokenId = 8,
        noTimestampsTokenId = 9,
        languageToId = mapOf("en" to 10),
        suppressTokens = setOf(2),
        beginSuppressTokens = setOf(3),
    )

    @Test
    fun forcedPrompt_matchesProductionPromptAndReferencePrefix() {
        val prompt = WhisperSplitDecoderTensors.forcedPrompt(
            generationConfig = config,
            language = WhisperLanguage.Forced("en"),
            referenceTokenPrefix = listOf(11, 12),
        )

        assertEquals(listOf(7, 10, 8, 9, 11, 12), prompt)
    }

    @Test
    fun forcedPrompt_rejectsOutOfVocabularyReferenceToken() {
        assertThrows(IllegalArgumentException::class.java) {
            WhisperSplitDecoderTensors.forcedPrompt(
                generationConfig = config,
                language = WhisperLanguage.Forced("en"),
                referenceTokenPrefix =
                    listOf(WhisperSplitDecoderTensors.VOCABULARY_SIZE),
            )
        }
    }

    @Test
    fun paddedTokens_keepsPromptAndFillsContextWithEos() {
        val padded = WhisperSplitDecoderTensors.paddedTokens(
            prompt = listOf(7, 10, 8, 9, 11),
            paddingToken = 6,
        )

        assertEquals(WhisperSplitDecoderTensors.TARGET_LENGTH, padded.size)
        assertArrayEquals(intArrayOf(7, 10, 8, 9, 11), padded.copyOfRange(0, 5))
        assertArrayEquals(
            IntArray(WhisperSplitDecoderTensors.TARGET_LENGTH - 5) { 6 },
            padded.copyOfRange(5, padded.size),
        )
    }

    @Test
    fun causalMask_isZeroOnAndBelowDiagonalAndNegativeAboveIt() {
        val mask = WhisperSplitDecoderTensors.causalMask()
        val width = WhisperSplitDecoderTensors.TARGET_LENGTH

        assertEquals(0.0f, mask[0], 0.0f)
        assertEquals(-1.0e9f, mask[1], 0.0f)
        assertEquals(0.0f, mask[width], 0.0f)
        assertEquals(0.0f, mask[width + 1], 0.0f)
        assertEquals(-1.0e9f, mask[width + 2], 0.0f)
        assertEquals(0.0f, mask.last(), 0.0f)
    }

    @Test
    fun activeHiddenRow_usesLastPromptPosition() {
        val width = WhisperSplitDecoderTensors.ENCODER_WIDTH
        val hidden = FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
        for (row in 0 until WhisperSplitDecoderTensors.TARGET_LENGTH) {
            for (column in 0 until width) {
                hidden[row * width + column] = row * 10_000f + column
            }
        }

        val active = WhisperSplitDecoderTensors.activeHiddenRow(
            decoderHidden = hidden,
            promptLength = 6,
        )

        assertEquals(width, active.size)
        assertEquals(50_000f, active.first(), 0.0f)
        assertEquals(51_279f, active.last(), 0.0f)
    }

    @Test
    fun activeHiddenRow_rejectsWrongCoreOutputShape() {
        assertThrows(IllegalArgumentException::class.java) {
            WhisperSplitDecoderTensors.activeHiddenRow(
                decoderHidden = FloatArray(1),
                promptLength = 1,
            )
        }
    }

    @Test
    fun projectionSignature_acceptsEitherSupportedSplitStage() {
        assertEquals("project", selectProjectionSignature(listOf("project")))
        assertEquals(
            "normalize_and_project",
            selectProjectionSignature(listOf("normalize_and_project")),
        )
    }

    @Test
    fun projectionSignature_rejectsMissingOrAmbiguousContracts() {
        assertThrows(WhisperModelContractException::class.java) {
            selectProjectionSignature(emptyList())
        }
        assertThrows(WhisperModelContractException::class.java) {
            selectProjectionSignature(listOf("project", "normalize_and_project"))
        }
    }

    @Test
    fun gpuOptionSelection_defaultsMatchExistingOpenClGpuCpuConfiguration() {
        val selection = selectWhisperSplitDecoderGpuOptions()

        assertEquals(
            listOf(Accelerator.GPU, Accelerator.CPU),
            selection.accelerators,
        )
        assertTrue(selection.constantTensorSharing)
        assertTrue(selection.allowSourceQuantizedFcConvOps)
        assertEquals(
            CompiledModel.GpuOptions.Backend.OPENCL,
            selection.backend,
        )
    }

    @Test
    fun gpuOptionSelection_strictModeSelectsOnlyGpuAndPreservesSwitches() {
        val selection = selectWhisperSplitDecoderGpuOptions(
            constantTensorSharing = false,
            allowSourceQuantizedFcConvOps = false,
            gpuBackend = WhisperSplitDecoderGpuBackend.WEBGPU,
            strictGpuOnly = true,
        )

        assertEquals(listOf(Accelerator.GPU), selection.accelerators)
        assertFalse(selection.constantTensorSharing)
        assertFalse(selection.allowSourceQuantizedFcConvOps)
        assertEquals(
            CompiledModel.GpuOptions.Backend.WEBGPU,
            selection.backend,
        )
    }

    @Test
    fun gpuOptionSelection_mapsEveryPublicBackendToLiteRt() {
        val mappings = mapOf(
            WhisperSplitDecoderGpuBackend.AUTOMATIC to
                CompiledModel.GpuOptions.Backend.AUTOMATIC,
            WhisperSplitDecoderGpuBackend.OPENCL to
                CompiledModel.GpuOptions.Backend.OPENCL,
            WhisperSplitDecoderGpuBackend.WEBGPU to
                CompiledModel.GpuOptions.Backend.WEBGPU,
            WhisperSplitDecoderGpuBackend.OPENGL to
                CompiledModel.GpuOptions.Backend.OPENGL,
        )

        assertEquals(
            WhisperSplitDecoderGpuBackend.entries.toSet(),
            mappings.keys,
        )
        mappings.forEach { (requested, expected) ->
            assertEquals(
                expected,
                selectWhisperSplitDecoderGpuOptions(gpuBackend = requested).backend,
            )
        }
    }
}
