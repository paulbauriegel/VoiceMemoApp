package com.example.voicememo.speaker

/**
 * One window-local speaker found by the segmentation model, with the voiceprint
 * of (part of) what that speaker said in the window. Independent of any voice
 * profile, so it can be stored and re-labelled later without the audio.
 */
internal class LocalSpeakerVoiceprint(
    val chunkIndex: Int,
    val localSpeaker: Int,
    /** First 16 kHz sample of the 10-second window. */
    val startSample: Int,
    /** Per segmentation frame: this speaker is talking. */
    val activeFrames: BooleanArray,
    val vector: FloatArray,
)

/**
 * First, profile-independent half of speaker labelling: segmentation of one
 * 10-second window, then one voiceprint per window-local speaker.
 *
 * Settings validated on a hand-marked recording (docs/transcription-benchmarks.md):
 * at most [MAX_VOICEPRINT_SECONDS] of each speaker's frames go through the
 * voice model, preferring frames where only that speaker talks.
 */
internal class SpeakerVoiceprintCollector(
    /** Up to WINDOW_SAMPLES of 16 kHz audio to log-probabilities `[FRAMES, CLASSES]`. */
    private val segment: (FloatArray) -> FloatArray,
    /** Speech samples to a unit-length voiceprint, or null when too short. */
    private val embed: (FloatArray) -> FloatArray?,
) {
    fun collect(chunkIndex: Int, startSample: Int, window: FloatArray): List<LocalSpeakerVoiceprint> {
        require(window.size <= PyannoteSegmenter.WINDOW_SAMPLES)
        val masks = SpeakerPowerset.activeSpeakers(segment(window))
        val frames = masks.size
        // Frames whose centre lies beyond the audio are padding, not speech.
        for (frame in 0 until frames) {
            if (frame * SHIFT + CENTRE >= window.size) masks[frame] = 0
        }
        val result = ArrayList<LocalSpeakerVoiceprint>(SpeakerPowerset.LOCAL_SPEAKERS)
        for (speaker in 0 until SpeakerPowerset.LOCAL_SPEAKERS) {
            val active = BooleanArray(frames) { SpeakerPowerset.isActive(masks[it], speaker) }
            val activeCount = active.count { it }
            if (activeCount < MIN_ACTIVE_FRAMES) continue
            var chosen = (0 until frames).filter { active[it] && !SpeakerPowerset.isOverlap(masks[it]) }
            if (chosen.size < MIN_ALONE_FRAMES) chosen = (0 until frames).filter { active[it] }
            if (chosen.size > MAX_VOICEPRINT_FRAMES) {
                val first = (chosen.size - MAX_VOICEPRINT_FRAMES) / 2
                chosen = chosen.subList(first, first + MAX_VOICEPRINT_FRAMES)
            }
            val speech = FloatArray(chosen.size * FRAME_SAMPLES)
            var length = 0
            for (frame in chosen) {
                val from = frame * SHIFT + CENTRE - SHIFT / 2
                val to = minOf(from + FRAME_SAMPLES, window.size)
                if (to <= from) continue
                window.copyInto(speech, length, from, to)
                length += to - from
            }
            if (length < CamPlusPlusEmbedder.MIN_SAMPLES) continue
            val vector = embed(if (length == speech.size) speech else speech.copyOf(length)) ?: continue
            result += LocalSpeakerVoiceprint(chunkIndex, speaker, startSample, active, vector)
        }
        return result
    }

    companion object {
        /** Windows are taken every 2.5 seconds, so each moment is seen four times. */
        const val HOP_SAMPLES = 40_000
        const val MAX_VOICEPRINT_SECONDS = 4
        private const val SHIFT = PyannoteSegmenter.FRAME_SHIFT_SAMPLES
        private const val CENTRE = PyannoteSegmenter.RECEPTIVE_FIELD_SAMPLES / 2
        /** Samples taken per frame: one shift around the frame centre. */
        private const val FRAME_SAMPLES = SHIFT + 1
        /** 0.25 s of activity; less is ignored. */
        private const val MIN_ACTIVE_FRAMES = 15
        /** 0.4 s alone; with less, overlapped frames are used too. */
        private const val MIN_ALONE_FRAMES = 24
        private const val MAX_VOICEPRINT_FRAMES =
            MAX_VOICEPRINT_SECONDS * PyannoteSegmenter.SAMPLE_RATE / SHIFT
    }
}
