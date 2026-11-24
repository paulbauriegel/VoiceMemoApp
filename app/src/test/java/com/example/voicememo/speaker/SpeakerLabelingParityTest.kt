package com.example.voicememo.speaker

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Checks the Kotlin labelling logic against the numpy/scipy reference in
 * scripts/generate-speaker-labeling-fixture.py.
 */
class SpeakerLabelingParityTest {
    private val lines = checkNotNull(javaClass.classLoader?.getResourceAsStream("speaker/labeling-parity.txt"))
        .bufferedReader().use { it.readLines() }

    private fun vector(line: String) = line.split(' ').map(String::toFloat).toFloatArray()

    @Test
    fun adaptationDecidesLikeTheReference() {
        val profileAt = lines.indexOfFirst { it.startsWith("profile ") }
        val profileCount = lines[profileAt].substringAfter(' ').toInt()
        val profile = lines.subList(profileAt + 1, profileAt + 1 + profileCount).map(::vector)
        var cases = 0
        lines.forEachIndexed { index, line ->
            if (!line.startsWith("case ")) return@forEachIndexed
            val (_, name, count, threshold, expected) = line.split(' ')
            val voiceprints = lines.subList(index + 1, index + 1 + count.toInt()).map(::vector)
            val decider = SpeakerDecision.adapt(voiceprints, profile)
            assertEquals("$name threshold", threshold.toFloat(), decider.threshold, 1e-4f)
            assertEquals(name, expected, voiceprints.joinToString("") { if (decider.isMe(it)) "1" else "0" })
            cases++
        }
        assertEquals(5, cases)
    }

    @Test
    fun timelineAndSegmentLabelsMatchTheReference() {
        val at = lines.indexOfFirst { it.startsWith("timeline ") }
        val (_, samples, count, expected) = lines[at].split(' ')
        val speakers = lines.subList(at + 1, at + 1 + count.toInt()).mapIndexed { index, line ->
            val (start, isMe, active) = line.split(' ')
            LocalSpeakerVoiceprint(
                chunkIndex = index,
                localSpeaker = 0,
                startSample = start.toInt(),
                activeFrames = BooleanArray(active.length) { active[it] == '1' },
                vector = floatArrayOf(if (isMe == "1") 1f else 0f),
            )
        }
        val timeline = SpeakerTimeline.build(speakers, samples.toInt()) { it[0] == 1f }
        val actual = SpeakerTimeline.decode(timeline.encode())
        assertEquals(expected.length, actual.frameCount)
        assertArrayEquals(timeline.encode(), actual.encode())
        val segmentsAt = lines.indexOfFirst { it.startsWith("segments ") }
        val segmentCount = lines[segmentsAt].substringAfter(' ').toInt()
        val labels = HashSet<SpeakerLabel>()
        for (line in lines.subList(segmentsAt + 1, segmentsAt + 1 + segmentCount)) {
            val (start, end, label) = line.split(' ')
            assertEquals(line, SpeakerLabel.valueOf(label), actual.labelFor(start.toLong(), end.toLong()))
            labels += SpeakerLabel.valueOf(label)
        }
        assertEquals(SpeakerLabel.entries.toSet(), labels)
        // Frame-exact check through a one-frame "segment" per 10 ms would need
        // five voiced frames; compare the run-length encoding with the reference instead.
        val reference = SpeakerTimeline.decode(runLength(expected))
        assertArrayEquals(reference.encode(), actual.encode())
    }

    private fun runLength(labels: String): ByteArray {
        val output = java.io.ByteArrayOutputStream()
        var index = 0
        while (index < labels.length) {
            var run = 1
            while (index + run < labels.length && labels[index + run] == labels[index]) run++
            output.write(labels[index] - '0')
            var remaining = run
            while (remaining >= 0x80) {
                output.write((remaining and 0x7F) or 0x80)
                remaining = remaining ushr 7
            }
            output.write(remaining)
            index += run
        }
        return output.toByteArray()
    }
}
