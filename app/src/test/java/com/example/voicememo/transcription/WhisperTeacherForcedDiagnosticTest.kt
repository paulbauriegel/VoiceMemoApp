package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WhisperTeacherForcedDiagnosticTest {
    private val config = WhisperGenerationConfig(
        eosTokenId = 6,
        decoderStartTokenId = 7,
        transcribeTokenId = 8,
        noTimestampsTokenId = 9,
        languageToId = mapOf("en" to 7),
        suppressTokens = setOf(2),
        beginSuppressTokens = setOf(3),
    )

    @Test
    fun ranking_matchesProductionSuppressionAndTieBreaking() {
        val logits = floatArrayOf(
            0f,
            1f,
            100f,
            90f,
            8f,
            8f,
            7f,
            110f,
            120f,
            130f,
        )

        val firstStep = WhisperTeacherForcedTokenRanking.forGpu(
            logits = logits,
            generationConfig = config,
            firstGeneratedToken = true,
        )
        val laterStep = WhisperTeacherForcedTokenRanking.forGpu(
            logits = logits,
            generationConfig = config,
            firstGeneratedToken = false,
        )

        assertEquals(4, firstStep.selectableTop1?.tokenId)
        assertEquals(5, firstStep.selectableTop2?.tokenId)
        assertEquals(0f, requireNotNull(firstStep.selectableMargin), 0f)
        assertEquals(3, laterStep.selectableTop1?.tokenId)
        assertEquals(4, laterStep.selectableTop2?.tokenId)
    }

    @Test
    fun rankingMirrorsCpuAndGpuNonFinitePolicies() {
        val logits = FloatArray(10) { Float.NEGATIVE_INFINITY }
        logits[1] = Float.POSITIVE_INFINITY

        val cpu = WhisperTeacherForcedTokenRanking.forCpu(
            logits = logits,
            generationConfig = config,
            firstGeneratedToken = false,
        )
        val gpu = WhisperTeacherForcedTokenRanking.forGpu(
            logits = logits,
            generationConfig = config,
            firstGeneratedToken = false,
        )

        assertEquals(1, cpu.selectableTop1?.tokenId)
        assertNull(gpu.selectableTop1)
        assertNull(gpu.selectableTop2)
        assertNull(gpu.selectableMargin)
    }
}
