package com.example.voicememo.speaker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeakerVoiceprintCollectorTest {
    private val frames = PyannoteSegmenter.FRAMES

    /** Log-probabilities where frame f has class `classOf(f)`. */
    private fun segmentation(classOf: (Int) -> Int) = FloatArray(frames * SpeakerPowerset.CLASSES) { index ->
        if (index % SpeakerPowerset.CLASSES == classOf(index / SpeakerPowerset.CLASSES)) -0.1f else -9f
    }

    @Test
    fun embedsEachSpeakerAndCapsTheAudio() {
        val embedded = ArrayList<Int>()
        val collector = SpeakerVoiceprintCollector(
            // speaker 0 for frames 0..399, both for 400..419, speaker 1 for 420..449, then silence
            segment = { segmentation { frame -> if (frame < 400) 1 else if (frame < 420) 4 else if (frame < 450) 2 else 0 } },
            embed = { samples -> embedded += samples.size; floatArrayOf(samples.size.toFloat()) },
        )
        val result = collector.collect(chunkIndex = 3, startSample = 120_000, window = FloatArray(160_000) { it.toFloat() })
        assertEquals(listOf(0, 1), result.map { it.localSpeaker })
        // Speaker 0: 400 frames alone, capped to 4 s = 237 frames of 271 samples.
        assertEquals(237 * 271, embedded[0])
        // Speaker 1: 30 frames alone (>= 0.4 s), so the overlapped frames are left out.
        assertEquals(30 * 271, embedded[1])
        assertEquals(420, result[0].activeFrames.count { it })
        assertEquals(50, result[1].activeFrames.count { it })
        assertEquals(120_000, result[0].startSample)
        assertEquals(3, result[0].chunkIndex)
    }

    @Test
    fun ignoresBriefSpeakersAndPadding() {
        val collector = SpeakerVoiceprintCollector(
            segment = { segmentation { frame -> if (frame < 10) 3 else 1 } },
            embed = { floatArrayOf(1f) },
        )
        // Only 5 s of audio: later frames are padding and must not count as speech.
        val result = collector.collect(0, 0, FloatArray(80_000))
        assertEquals(listOf(0), result.map { it.localSpeaker })
        val active = result.single().activeFrames
        assertTrue(active[100])
        assertFalse(active[frames - 1])
        assertFalse(active[(80_000 - 495) / 270 + 1])
    }
}
