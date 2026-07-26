package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperSpeculativeDiagnosticTest {
    @Test
    fun marginDiagnosticsRemainOptionalForExistingSpeculativeCallers() {
        val proposal = WhisperGpuProposedTokenBlock(
            tokens = listOf(10),
            decoderInvocations = 1,
            inferenceMillis = 1L,
        )
        val verification = WhisperCpuVerifiedTokenBlock(
            proposedTokens = listOf(10),
            cpuSelectedTokens = listOf(10),
            inferenceMillis = 1L,
        )

        assertTrue(proposal.gpuSelectableTop1Margins.isEmpty())
        assertTrue(verification.cpuSelectableTop1Margins.isEmpty())
        assertTrue(verification.proposalMatchesCpuTop1.isEmpty())
    }

    @Test
    fun acceptsEntireMatchingBlockWithoutCorrection() {
        val result = WhisperSpeculativeTokenAcceptance.accept(
            verification = verification(
                proposed = listOf(10, 20, 30),
                cpu = listOf(10, 20, 30),
            ),
            eosTokenId = EOS,
        )

        assertEquals(listOf(10, 20, 30), result.acceptedTokens)
        assertEquals(3, result.matchingProposalCount)
        assertNull(result.correctedToken)
        assertFalse(result.reachedEos)
        assertTrue(result.allProposalsAccepted)
    }

    @Test
    fun acceptsMatchingPrefixAndCpuCorrectionOnly() {
        val result = WhisperSpeculativeTokenAcceptance.accept(
            verification = verification(
                proposed = listOf(10, 99, 77),
                cpu = listOf(10, 20, 88),
            ),
            eosTokenId = EOS,
        )

        assertEquals(listOf(10, 20), result.acceptedTokens)
        assertEquals(1, result.matchingProposalCount)
        assertEquals(20, result.correctedToken)
        assertFalse(result.reachedEos)
        assertFalse(result.allProposalsAccepted)
    }

    @Test
    fun matchingCpuEosTerminatesWithoutAppendingEos() {
        val result = WhisperSpeculativeTokenAcceptance.accept(
            verification = verification(
                proposed = listOf(10, EOS),
                cpu = listOf(10, EOS),
            ),
            eosTokenId = EOS,
        )

        assertEquals(listOf(10), result.acceptedTokens)
        assertEquals(2, result.matchingProposalCount)
        assertNull(result.correctedToken)
        assertTrue(result.reachedEos)
        assertTrue(result.allProposalsAccepted)
    }

    @Test
    fun cpuEosCorrectionTerminatesAndDiscardsLaterRows() {
        val result = WhisperSpeculativeTokenAcceptance.accept(
            verification = verification(
                proposed = listOf(10, 99, 77),
                cpu = listOf(10, EOS, 88),
            ),
            eosTokenId = EOS,
        )

        assertEquals(listOf(10), result.acceptedTokens)
        assertEquals(1, result.matchingProposalCount)
        assertEquals(EOS, result.correctedToken)
        assertTrue(result.reachedEos)
        assertFalse(result.allProposalsAccepted)
    }

    private fun verification(
        proposed: List<Int>,
        cpu: List<Int>,
    ) = WhisperCpuVerifiedTokenBlock(
        proposedTokens = proposed,
        cpuSelectedTokens = cpu,
        inferenceMillis = 1L,
    )

    private companion object {
        const val EOS = 50_257
    }
}
