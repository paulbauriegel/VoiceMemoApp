package com.example.voicememo.speaker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class CamPlusPlusEmbedderTest {
    /** [frames] feature frames whose every value is the frame number. */
    private fun features(frames: Int) =
        FloatArray(frames * KaldiFbank.MEL_BINS) { (it / KaldiFbank.MEL_BINS).toFloat() }

    private fun frameNumbers(features: FloatArray) =
        FloatArray(features.size / KaldiFbank.MEL_BINS) { features[it * KaldiFbank.MEL_BINS] }

    @Test
    fun repeatsFramesFromTheStartUpToTheNextMultiple() {
        val padded = CamPlusPlusEmbedder.padFrames(features(3), multiple = 4)
        assertEquals(4 * KaldiFbank.MEL_BINS, padded.size)
        assertArrayEquals(floatArrayOf(0f, 1f, 2f, 0f), frameNumbers(padded), 0f)
        // Every value of a frame is copied, not only the first.
        assertEquals(0f, padded[4 * KaldiFbank.MEL_BINS - 1], 0f)
    }

    @Test
    fun repeatsShortInputSeveralTimes() {
        val padded = CamPlusPlusEmbedder.padFrames(features(2), multiple = 5)
        assertArrayEquals(floatArrayOf(0f, 1f, 0f, 1f, 0f), frameNumbers(padded), 0f)
    }

    @Test
    fun padsTheQuarterSecondMinimumToOneSegment() {
        // 0.25 s is 23 frames; 1.5 s is 148; both must reach a whole pooling segment.
        assertEquals(200, CamPlusPlusEmbedder.padFrames(features(23), CamPlusPlusEmbedder.FRAME_PADDING).size / KaldiFbank.MEL_BINS)
        assertEquals(200, CamPlusPlusEmbedder.padFrames(features(148), CamPlusPlusEmbedder.FRAME_PADDING).size / KaldiFbank.MEL_BINS)
        assertEquals(400, CamPlusPlusEmbedder.padFrames(features(201), CamPlusPlusEmbedder.FRAME_PADDING).size / KaldiFbank.MEL_BINS)
    }

    @Test
    fun leavesWholeSegmentsAndDisabledPaddingAlone() {
        val whole = features(400)
        assertSame(whole, CamPlusPlusEmbedder.padFrames(whole, CamPlusPlusEmbedder.FRAME_PADDING))
        val odd = features(7)
        assertSame(odd, CamPlusPlusEmbedder.padFrames(odd, 0))
    }
}
