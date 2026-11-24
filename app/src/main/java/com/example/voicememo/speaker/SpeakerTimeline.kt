package com.example.voicememo.speaker

import java.io.ByteArrayOutputStream
import kotlin.math.PI
import kotlin.math.cos

enum class SpeakerLabel { ME, OTHER, MIXED, UNKNOWN }

/**
 * Who is talking, on a 10 ms grid over one recording: nobody, the enrolled
 * person, or someone else. Transcript segments read their label from it, so it
 * stays valid when segments are merged or edited.
 */
class SpeakerTimeline internal constructor(private val frames: ByteArray) {
    val frameCount: Int get() = frames.size

    /**
     * Label for a transcript segment: ME or OTHER when at least three quarters
     * of its voiced time is one side, MIXED in between, UNKNOWN with under
     * 50 ms of detected speech.
     */
    fun labelFor(startMillis: Long, endMillis: Long): SpeakerLabel {
        val first = (startMillis / FRAME_MILLIS).toInt().coerceIn(0, frames.size)
        val last = maxOf((endMillis / FRAME_MILLIS).toInt(), first + 1).coerceIn(0, frames.size)
        var me = 0
        var other = 0
        for (index in first until last) {
            when (frames[index]) {
                ME -> me++
                OTHER -> other++
            }
        }
        val voiced = me + other
        if (voiced < MIN_VOICED_FRAMES) return SpeakerLabel.UNKNOWN
        val share = me.toFloat() / voiced
        return when {
            share >= CLEAR_SHARE -> SpeakerLabel.ME
            share <= 1 - CLEAR_SHARE -> SpeakerLabel.OTHER
            else -> SpeakerLabel.MIXED
        }
    }

    /** Run-length encoding: label byte, then the run length as a varint. */
    fun encode(): ByteArray {
        val output = ByteArrayOutputStream()
        var index = 0
        while (index < frames.size) {
            val label = frames[index]
            var run = 1
            while (index + run < frames.size && frames[index + run] == label) run++
            output.write(label.toInt())
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

    companion object {
        const val FRAME_MILLIS = 10L
        internal const val NONE: Byte = 0
        internal const val ME: Byte = 1
        internal const val OTHER: Byte = 2
        private const val MIN_VOICED_FRAMES = 5
        private const val CLEAR_SHARE = 0.75f
        private const val SAMPLES_PER_FRAME = PyannoteSegmenter.SAMPLE_RATE / 100

        fun decode(encoded: ByteArray): SpeakerTimeline {
            val output = ByteArrayOutputStream()
            var index = 0
            while (index < encoded.size) {
                val label = encoded[index++]
                require(label in NONE..OTHER) { "Unknown speaker label $label." }
                var run = 0
                var shift = 0
                while (true) {
                    val part = encoded[index++].toInt() and 0xFF
                    run = run or ((part and 0x7F) shl shift)
                    if (part < 0x80) break
                    shift += 7
                }
                repeat(run) { output.write(label.toInt()) }
            }
            return SpeakerTimeline(output.toByteArray())
        }

        /**
         * Lets every window-local speaker vote on the 10 ms frames it talks in.
         * Windows overlap, so each moment gets several votes; votes from the
         * middle of a window count more than those at its edges.
         */
        internal fun build(
            speakers: List<LocalSpeakerVoiceprint>,
            sampleCount: Int,
            isMe: (FloatArray) -> Boolean,
        ): SpeakerTimeline {
            val frameCount = (sampleCount + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME + 10
            val me = FloatArray(frameCount)
            val other = FloatArray(frameCount)
            val shift = PyannoteSegmenter.FRAME_SHIFT_SAMPLES
            val centre = PyannoteSegmenter.RECEPTIVE_FIELD_SAMPLES / 2
            for (speaker in speakers) {
                val votes = if (isMe(speaker.vector)) me else other
                val frames = speaker.activeFrames
                for (frame in frames.indices) {
                    if (!frames[frame]) continue
                    val index = ((speaker.startSample + frame * shift + centre).toLong() * 100 /
                        PyannoteSegmenter.SAMPLE_RATE).toInt()
                    if (index >= frameCount) continue
                    votes[index] += windowWeight(frame, frames.size)
                }
            }
            return SpeakerTimeline(
                ByteArray(frameCount) { index ->
                    when {
                        me[index] + other[index] <= 0f -> NONE
                        me[index] >= other[index] -> ME
                        else -> OTHER
                    }
                },
            )
        }

        /** Hann window over the frames of one segmentation window, never quite zero. */
        private fun windowWeight(frame: Int, frames: Int): Float =
            (0.5 - 0.5 * cos(2.0 * PI * (frame + 1) / (frames + 1))).toFloat() + 1e-3f
    }
}
