package com.example.voicememo.transcription

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtDecoderBoundaryDiagnosticTest {
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
    fun stagesExposeExactGeneratedSignatureAndBackendRoutes() {
        assertEquals(
            "decode_tail_after_position_add",
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD.tailSignature,
        )
        assertEquals(
            "hidden_after_position_add",
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD.boundaryInputName,
        )
        assertEquals(
            WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD.prefixBackend,
        )
        assertEquals(
            WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD.tailBackend,
        )

        val blockStages = listOf(
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD,
            WhisperDecoderBoundaryStage.AFTER_CROSS_ATTENTION,
            WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            WhisperDecoderBoundaryStage.AFTER_2_BLOCKS,
            WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
        )
        blockStages.forEach { stage ->
            assertTrue(stage.tailSignature.startsWith("decode_tail_after_"))
            assertEquals(
                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
                stage.prefixBackend,
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                stage.tailBackend,
            )
        }
        assertEquals(
            "decode_tail_after_3_blocks",
            WhisperDecoderBoundaryStage.AFTER_3_BLOCKS.tailSignature,
        )
        assertEquals(
            "hidden_after_3_decoder_blocks",
            WhisperDecoderBoundaryStage.AFTER_3_BLOCKS.boundaryInputName,
        )
    }

    @Test
    fun stagesResolveFromDiagnosticArgumentIds() {
        WhisperDecoderBoundaryStage.entries.forEach { stage ->
            assertEquals(stage, WhisperDecoderBoundaryStage.fromId(stage.id))
        }
        assertNull(WhisperDecoderBoundaryStage.fromId("post-layernorm"))
        assertNull(WhisperDecoderBoundaryStage.fromId("unknown"))
    }

    @Test
    fun executionModesRouteBothSidesWithoutChangingTheStage() {
        val routed = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED
        val reversed =
            WhisperDecoderBoundaryExecutionMode.CPU_PREFIX_GPU_TAIL
        val control = WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL

        WhisperDecoderBoundaryStage.entries.forEach { stage ->
            assertEquals(
                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
                routed.prefixBackend(stage),
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                routed.tailBackend(stage),
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                reversed.prefixBackend(stage),
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.GPU_COMPILED_MODEL,
                reversed.tailBackend(stage),
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                control.prefixBackend(stage),
            )
            assertEquals(
                WhisperSplitDecoderCoreBackend.CPU_XNNPACK,
                control.tailBackend(stage),
            )
        }
    }

    @Test
    fun blockBoundaryAssetsPointAtTheirCompactSharedModelsAndCpuProjection() {
        val expectedSharedModels = mapOf(
            WhisperDecoderBoundaryStage.AFTER_POSITION_ADD to
                "whisper_large_v3_turbo_decode_shared_after_position_add.tflite",
            WhisperDecoderBoundaryStage.AFTER_CROSS_ATTENTION to
                "whisper_large_v3_turbo_decode_shared_after_cross_attention.tflite",
            WhisperDecoderBoundaryStage.AFTER_1_BLOCK to
                "whisper_large_v3_turbo_decode_shared_after_1_block.tflite",
            WhisperDecoderBoundaryStage.AFTER_2_BLOCKS to
                "whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite",
            WhisperDecoderBoundaryStage.AFTER_3_BLOCKS to
                "whisper_large_v3_turbo_decode_shared_after_3_blocks.tflite",
        )

        expectedSharedModels.forEach { (stage, expectedFileName) ->
            val assets = WhisperDecoderBoundaryAssets.forStage(stage)
            assertEquals(
                "whisper-decoder-split/$expectedFileName",
                assets.sharedDecoder,
            )
            assertEquals(
                "whisper-decoder-split/" +
                    "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite",
                assets.activeRowNormalizeAndProject,
            )
        }
    }

    @Test
    fun teacherForcedStepPassesFullBoundaryAndProjectsOnlyActivePreNormRow() {
        val activeRowIndex = 5
        val activeStart =
            activeRowIndex * WhisperSplitDecoderTensors.ENCODER_WIDTH
        val boundaryOutput =
            FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS).also {
                it[activeStart] = 101f
                it[activeStart + WhisperSplitDecoderTensors.ENCODER_WIDTH - 1] =
                    102f
            }
        val preLayerNorm =
            FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS).also {
                it[activeStart] = 201f
                it[activeStart + WhisperSplitDecoderTensors.ENCODER_WIDTH - 1] =
                    202f
            }
        var observedTokens: IntArray? = null
        var tailSawExactBoundary = false
        var projectedExactActiveRow = false
        val prefix = object : WhisperDecoderBoundaryPrefix {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                observedTokens = tokens.copyOf()
                return boundaryOutput
            }

            override fun close() = Unit
        }
        val tail = object : WhisperDecoderBoundaryTail {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                tailSawExactBoundary = boundary === boundaryOutput
                return preLayerNorm
            }

            override fun close() = Unit
        }
        val projection = object : WhisperDecoderBoundaryProjection {
            override fun run(
                activePreLayerNorm: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                projectedExactActiveRow =
                    activePreLayerNorm.first() == 201f &&
                        activePreLayerNorm.last() == 202f
                return FloatArray(WhisperSplitDecoderTensors.VOCABULARY_SIZE).also {
                    it[4] = 9f
                    it[5] = 10f
                }
            }

            override fun close() = Unit
        }
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = prefix,
            tail = tail,
            projection = projection,
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
            executionMode = WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL,
        )

        val result = diagnostic.teacherForcedStep(
            encoderState =
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
            language = WhisperLanguage.Forced("en"),
            referenceTokenPrefix = listOf(11, 12),
        )

        assertEquals(6, result.promptLength)
        assertArrayEquals(
            intArrayOf(7, 10, 8, 9, 11, 12),
            checkNotNull(observedTokens).copyOfRange(0, 6),
        )
        assertTrue(
            checkNotNull(observedTokens)
                .copyOfRange(6, WhisperSplitDecoderTensors.TARGET_LENGTH)
                .all { it == 6 },
        )
        assertTrue(tailSawExactBoundary)
        assertTrue(projectedExactActiveRow)
        assertEquals(101f, result.activeBoundaryRow.first(), 0f)
        assertEquals(102f, result.activeBoundaryRow.last(), 0f)
        assertEquals(201f, result.activePreLayerNormRow.first(), 0f)
        assertEquals(202f, result.activePreLayerNormRow.last(), 0f)
        assertEquals(5, result.rankedLogits.selectableTop1?.tokenId)
        assertEquals(4, result.rankedLogits.selectableTop2?.tokenId)
        assertTrue(result.prefixMillis >= 0L)
        assertTrue(result.tailMillis >= 0L)
        assertTrue(result.projectionMillis >= 0L)
        diagnostic.close()
    }

    @Test
    fun closeIsIdempotentAndClosedRuntimeRejectsInference() {
        var prefixClosed = 0
        var tailClosed = 0
        var projectionClosed = 0
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = noOpPrefix { prefixClosed++ },
            tail = noOpTail { tailClosed++ },
            projection = noOpProjection { projectionClosed++ },
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
            executionMode = WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL,
        )

        diagnostic.close()
        diagnostic.close()

        assertEquals(1, prefixClosed)
        assertEquals(1, tailClosed)
        assertEquals(1, projectionClosed)
        assertThrows(IllegalStateException::class.java) {
            diagnostic.teacherForcedStep(
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
                WhisperLanguage.Forced("en"),
                emptyList(),
            )
        }
    }

    @Test
    fun cancellationStopsBeforeAnyBoundaryInvocation() {
        var invoked = false
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = object : WhisperDecoderBoundaryPrefix {
                override fun run(
                    encoderState: FloatArray,
                    tokens: IntArray,
                    shouldCancel: () -> Boolean,
                ): FloatArray {
                    invoked = true
                    return FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)
                }

                override fun close() = Unit
            },
            tail = noOpTail(),
            projection = noOpProjection(),
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_3_BLOCKS,
            executionMode = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
        )

        assertThrows(WhisperCancelledException::class.java) {
            diagnostic.teacherForcedStep(
                encoderState =
                    FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
                language = WhisperLanguage.Forced("en"),
                referenceTokenPrefix = emptyList(),
                shouldCancel = { true },
            )
        }
        assertFalse(invoked)
        diagnostic.close()
    }

    @Test
    fun speculativeProposalUsesCurrentCausalRowAndStopsAtEos() {
        val reusedBoundary = decoderHiddenWithRowMarkers()
        val reusedPreLayerNorm = decoderHiddenWithRowMarkers()
        val observedTokenInputs = mutableListOf<IntArray>()
        var tailInvocations = 0
        val prefix = object : WhisperDecoderBoundaryPrefix {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                observedTokenInputs += tokens.copyOf()
                return reusedBoundary
            }

            override fun close() = Unit
        }
        val tail = object : WhisperDecoderBoundaryTail,
            WhisperDecoderBoundaryCpuVerifier {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                assertSame(reusedBoundary, boundary)
                tailInvocations++
                return reusedPreLayerNorm
            }

            override fun runFullDecoder(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ) = reusedPreLayerNorm

            override fun close() = Unit
        }
        val projectedRows = mutableListOf<Int>()
        val projection = marginAwareRowSelectingProjection(
            projectedRows = projectedRows,
            tokenForRow = { row ->
                when (row) {
                    3 -> 4
                    4 -> config.eosTokenId
                    else -> error("Unexpected proposal row $row")
                }
            },
            marginForRow = { row ->
                when (row) {
                    3 -> 0.75f
                    4 -> 1.25f
                    else -> error("Unexpected proposal row $row")
                }
            },
        )
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = prefix,
            tail = tail,
            projection = projection,
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            executionMode = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
        )

        val result = diagnostic.proposeTokenBlock(
            encoderState =
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
            language = WhisperLanguage.Forced("en"),
            verifiedTokenPrefix = emptyList(),
            maximumTokenCount = 4,
        )

        assertEquals(listOf(4, config.eosTokenId), result.tokens)
        assertEquals(2, result.decoderInvocations)
        assertTrue(result.inferenceMillis >= 0L)
        assertEquals(listOf(0.75f, 1.25f), result.gpuSelectableTop1Margins)
        assertEquals(listOf(3, 4), projectedRows)
        assertEquals(2, tailInvocations)
        assertArrayEquals(
            intArrayOf(7, 10, 8, 9),
            observedTokenInputs[0].copyOfRange(0, 4),
        )
        assertArrayEquals(
            intArrayOf(7, 10, 8, 9, 4),
            observedTokenInputs[1].copyOfRange(0, 5),
        )
        assertTrue(
            observedTokenInputs[1]
                .copyOfRange(5, WhisperSplitDecoderTensors.TARGET_LENGTH)
                .all { it == config.eosTokenId },
        )
        diagnostic.close()
    }

    @Test
    fun speculativeVerificationUsesSharedCpuTailOnceAndFirstMismatchWins() {
        val reusablePreLayerNorm = decoderHiddenWithRowMarkers()
        var routedPrefixInvocations = 0
        var routedTailInvocations = 0
        var fullCpuInvocations = 0
        var sharedTailCloseCount = 0
        var observedCpuTokens: IntArray? = null
        val prefix = object : WhisperDecoderBoundaryPrefix {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                routedPrefixInvocations++
                return decoderHiddenWithRowMarkers()
            }

            override fun close() = Unit
        }
        val sharedCpuTail = object : WhisperDecoderBoundaryTail,
            WhisperDecoderBoundaryCpuVerifier {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                routedTailInvocations++
                return reusablePreLayerNorm
            }

            override fun runFullDecoder(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                fullCpuInvocations++
                observedCpuTokens = tokens.copyOf()
                return reusablePreLayerNorm
            }

            override fun close() {
                sharedTailCloseCount++
            }
        }
        val projectedRows = mutableListOf<Int>()
        val projection = marginAwareRowSelectingProjection(
            projectedRows = projectedRows,
            tokenForRow = { row ->
                when (row) {
                    4 -> 4
                    5 -> 5
                    6 -> config.eosTokenId
                    else -> error("Unexpected verification row $row")
                }
            },
            marginForRow = { row ->
                when (row) {
                    4 -> 0.5f
                    5 -> 0.25f
                    6 -> 0.125f
                    else -> error("Unexpected verification row $row")
                }
            },
        )
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = prefix,
            tail = sharedCpuTail,
            projection = projection,
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            executionMode = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
        )

        val verification = diagnostic.verifyProposedTokenBlock(
            encoderState =
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
            language = WhisperLanguage.Forced("en"),
            verifiedTokenPrefix = listOf(11),
            proposedTokens = listOf(4, 20, 21),
        )
        val acceptance = WhisperSpeculativeTokenAcceptance.accept(
            verification = verification,
            eosTokenId = config.eosTokenId,
        )

        assertEquals(listOf(4, 20, 21), verification.proposedTokens)
        assertEquals(listOf(4, 5, config.eosTokenId), verification.cpuSelectedTokens)
        assertEquals(
            listOf(0.5f, 0.25f, 0.125f),
            verification.cpuSelectableTop1Margins,
        )
        assertEquals(
            listOf(true, false, false),
            verification.proposalMatchesCpuTop1,
        )
        assertTrue(verification.inferenceMillis >= 0L)
        assertEquals(0, routedPrefixInvocations)
        assertEquals(0, routedTailInvocations)
        assertEquals(1, fullCpuInvocations)
        assertEquals(listOf(4, 5, 6), projectedRows)
        assertArrayEquals(
            intArrayOf(7, 10, 8, 9, 11, 4, 20, 21),
            checkNotNull(observedCpuTokens).copyOfRange(0, 8),
        )
        assertTrue(
            checkNotNull(observedCpuTokens)
                .copyOfRange(8, WhisperSplitDecoderTensors.TARGET_LENGTH)
                .all { it == config.eosTokenId },
        )
        assertEquals(listOf(4, 5), acceptance.acceptedTokens)
        assertEquals(1, acceptance.matchingProposalCount)
        assertEquals(5, acceptance.correctedToken)
        assertFalse(acceptance.reachedEos)
        assertFalse(acceptance.allProposalsAccepted)

        diagnostic.close()
        diagnostic.close()
        assertEquals(1, sharedTailCloseCount)
    }

    @Test
    fun reverseSpeculativeVerificationReusesSharedCpuPrefix() {
        val reusablePreLayerNorm = decoderHiddenWithRowMarkers()
        var routedPrefixInvocations = 0
        var routedTailInvocations = 0
        var fullCpuInvocations = 0
        var sharedPrefixCloseCount = 0
        val sharedCpuPrefix = object : WhisperDecoderBoundaryPrefix,
            WhisperDecoderBoundaryCpuVerifier {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                routedPrefixInvocations++
                return decoderHiddenWithRowMarkers()
            }

            override fun runFullDecoder(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                fullCpuInvocations++
                return reusablePreLayerNorm
            }

            override fun close() {
                sharedPrefixCloseCount++
            }
        }
        val gpuTail = object : WhisperDecoderBoundaryTail {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                routedTailInvocations++
                return reusablePreLayerNorm
            }

            override fun close() = Unit
        }
        val projectedRows = mutableListOf<Int>()
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = sharedCpuPrefix,
            tail = gpuTail,
            projection = rowSelectingProjection(projectedRows) { 4 },
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            executionMode =
                WhisperDecoderBoundaryExecutionMode.CPU_PREFIX_GPU_TAIL,
        )

        val verification = diagnostic.verifyProposedTokenBlock(
            encoderState =
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
            language = WhisperLanguage.Forced("en"),
            verifiedTokenPrefix = emptyList(),
            proposedTokens = listOf(4),
        )

        assertEquals(listOf(4), verification.cpuSelectedTokens)
        assertEquals(listOf(true), verification.proposalMatchesCpuTop1)
        assertEquals(listOf(3), projectedRows)
        assertEquals(0, routedPrefixInvocations)
        assertEquals(0, routedTailInvocations)
        assertEquals(1, fullCpuInvocations)
        diagnostic.close()
        diagnostic.close()
        assertEquals(1, sharedPrefixCloseCount)
    }

    @Test
    fun speculativeBlockRespectsExactDecoderContextLimit() {
        val preLayerNorm = decoderHiddenWithRowMarkers()
        var proposalInvocations = 0
        val prefix = object : WhisperDecoderBoundaryPrefix {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                proposalInvocations++
                return preLayerNorm
            }

            override fun close() = Unit
        }
        val sharedCpuTail = object : WhisperDecoderBoundaryTail,
            WhisperDecoderBoundaryCpuVerifier {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ) = preLayerNorm

            override fun runFullDecoder(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ) = preLayerNorm

            override fun close() = Unit
        }
        val diagnostic = LiteRtDecoderBoundaryDiagnostic(
            prefix = prefix,
            tail = sharedCpuTail,
            projection = rowSelectingProjection(mutableListOf()) { 4 },
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            executionMode = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
        )
        val maximumVerifiedPrefix =
            List(WhisperSplitDecoderTensors.TARGET_LENGTH - 5) { 11 }

        val proposal = diagnostic.proposeTokenBlock(
            encoderState =
                FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
            language = WhisperLanguage.Forced("en"),
            verifiedTokenPrefix = maximumVerifiedPrefix,
            maximumTokenCount = 4,
        )

        assertEquals(1, proposal.tokens.size)
        assertEquals(1, proposalInvocations)
        assertThrows(IllegalArgumentException::class.java) {
            diagnostic.verifyProposedTokenBlock(
                encoderState =
                    FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS),
                language = WhisperLanguage.Forced("en"),
                verifiedTokenPrefix = maximumVerifiedPrefix,
                proposedTokens = listOf(4, 5),
            )
        }
        diagnostic.close()
    }

    @Test
    fun speculativeMethodsRejectNonRoutedOrDifferentBoundaryStages() {
        val encoderState =
            FloatArray(WhisperSplitDecoderTensors.ENCODER_STATE_FLOATS)
        val wrongStage = LiteRtDecoderBoundaryDiagnostic(
            prefix = noOpPrefix(),
            tail = noOpTail(),
            projection = noOpProjection(),
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_2_BLOCKS,
            executionMode = WhisperDecoderBoundaryExecutionMode.STAGE_ROUTED,
        )
        val cpuControl = LiteRtDecoderBoundaryDiagnostic(
            prefix = noOpPrefix(),
            tail = noOpTail(),
            projection = noOpProjection(),
            generationConfig = config,
            stage = WhisperDecoderBoundaryStage.AFTER_1_BLOCK,
            executionMode = WhisperDecoderBoundaryExecutionMode.ALL_CPU_CONTROL,
        )

        assertThrows(IllegalStateException::class.java) {
            wrongStage.proposeTokenBlock(
                encoderState,
                WhisperLanguage.Forced("en"),
                emptyList(),
                1,
            )
        }
        assertThrows(IllegalStateException::class.java) {
            cpuControl.proposeTokenBlock(
                encoderState,
                WhisperLanguage.Forced("en"),
                emptyList(),
                1,
            )
        }
        wrongStage.close()
        cpuControl.close()
    }

    private fun decoderHiddenWithRowMarkers(): FloatArray =
        FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS).also { hidden ->
            repeat(WhisperSplitDecoderTensors.TARGET_LENGTH) { row ->
                hidden[row * WhisperSplitDecoderTensors.ENCODER_WIDTH] = row.toFloat()
            }
        }

    private fun rowSelectingProjection(
        projectedRows: MutableList<Int>,
        tokenForRow: (Int) -> Int,
    ): WhisperDecoderBoundaryProjection =
        object : WhisperDecoderBoundaryProjection {
            override fun run(
                activePreLayerNorm: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                val row = activePreLayerNorm.first().toInt()
                projectedRows += row
                return FloatArray(WhisperSplitDecoderTensors.VOCABULARY_SIZE) {
                    Float.NEGATIVE_INFINITY
                }.also { logits ->
                    logits[tokenForRow(row)] = 1f
                }
            }

            override fun close() = Unit
        }

    private fun marginAwareRowSelectingProjection(
        projectedRows: MutableList<Int>,
        tokenForRow: (Int) -> Int,
        marginForRow: (Int) -> Float,
    ): WhisperDecoderBoundaryProjection =
        object : WhisperDecoderBoundaryProjection {
            override fun run(
                activePreLayerNorm: FloatArray,
                shouldCancel: () -> Boolean,
            ): FloatArray {
                val row = activePreLayerNorm.first().toInt()
                projectedRows += row
                val selectedToken = tokenForRow(row)
                val runnerUpToken = if (selectedToken == 0) 1 else 0
                return FloatArray(
                    WhisperSplitDecoderTensors.VOCABULARY_SIZE,
                ) {
                    -10f
                }.also { logits ->
                    logits[runnerUpToken] = 1f
                    logits[selectedToken] = 1f + marginForRow(row)
                }
            }

            override fun close() = Unit
        }

    private fun noOpPrefix(
        onClose: () -> Unit = {},
    ): WhisperDecoderBoundaryPrefix =
        object : WhisperDecoderBoundaryPrefix {
            override fun run(
                encoderState: FloatArray,
                tokens: IntArray,
                shouldCancel: () -> Boolean,
            ) = FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

            override fun close() = onClose()
        }

    private fun noOpTail(
        onClose: () -> Unit = {},
    ): WhisperDecoderBoundaryTail =
        object : WhisperDecoderBoundaryTail {
            override fun run(
                encoderState: FloatArray,
                boundary: FloatArray,
                shouldCancel: () -> Boolean,
            ) = FloatArray(WhisperSplitDecoderTensors.DECODER_HIDDEN_FLOATS)

            override fun close() = onClose()
        }

    private fun noOpProjection(
        onClose: () -> Unit = {},
    ): WhisperDecoderBoundaryProjection =
        object : WhisperDecoderBoundaryProjection {
            override fun run(
                activePreLayerNorm: FloatArray,
                shouldCancel: () -> Boolean,
            ) = FloatArray(WhisperSplitDecoderTensors.VOCABULARY_SIZE)

            override fun close() = onClose()
        }
}
