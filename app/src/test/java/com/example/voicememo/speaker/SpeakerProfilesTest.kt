package com.example.voicememo.speaker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SpeakerProfilesTest {
    /** A window-local speaker talking from [fromSeconds] to [toSeconds] within a window starting at [windowSeconds]. */
    private fun speaker(chunk: Int, windowSeconds: Int, fromSeconds: Double, toSeconds: Double, vararg vector: Float) =
        LocalSpeakerVoiceprint(
            chunkIndex = chunk,
            localSpeaker = 0,
            startSample = windowSeconds * 16_000,
            activeFrames = BooleanArray(PyannoteSegmenter.FRAMES) { frame ->
                val seconds = windowSeconds + (frame * 270 + 495) / 16_000.0
                seconds >= fromSeconds && seconds < toSeconds
            },
            vector = vector,
        )

    @Test
    fun usesTheSpeakersHeardInsideTheSegment() {
        val speakers = listOf(
            speaker(0, 0, 2.0, 6.0, 1f, 0f),       // talks through the segment
            speaker(1, 0, 4.9, 5.0, 0f, 1f),       // 0.1 s inside: ignored
            speaker(2, 2, 3.0, 5.5, 0.6f, 0.8f),   // overlapping window, same stretch
            speaker(3, 10, 12.0, 15.0, 0f, -1f),   // elsewhere
        )
        val vector = SpeakerProfiles.segmentVoiceprint(speakers, startMillis = 3_000, endMillis = 5_000)
        // centroid of (1, 0) and (0.6, 0.8), normalised
        assertArrayEquals(floatArrayOf(0.8944272f, 0.4472136f), vector, 1e-5f)
    }

    @Test
    fun perWindowOnlyTheSpeakerTalkingMostCounts() {
        val speakers = listOf(
            speaker(0, 0, 1.0, 2.0, 1f, 0f),
            LocalSpeakerVoiceprint(0, 1, 0, speaker(0, 0, 1.0, 4.0, 0f).activeFrames, floatArrayOf(0f, 1f)),
        )
        assertArrayEquals(
            floatArrayOf(0f, 1f),
            SpeakerProfiles.segmentVoiceprint(speakers, startMillis = 1_000, endMillis = 4_000),
            1e-6f,
        )
    }

    @Test
    fun tooShortASegmentGivesNoVoiceprint() {
        val speakers = listOf(speaker(0, 0, 0.0, 9.0, 1f, 0f))
        assertNull(SpeakerProfiles.segmentVoiceprint(speakers, startMillis = 1_000, endMillis = 1_300))
    }
}
