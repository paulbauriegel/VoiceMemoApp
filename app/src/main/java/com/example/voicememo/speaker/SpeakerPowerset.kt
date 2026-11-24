package com.example.voicememo.speaker

/**
 * pyannote segmentation-3.0 predicts one of seven "powerset" classes per frame:
 * nobody, one of three window-local speakers, or one of the three pairs of them.
 */
internal object SpeakerPowerset {
    const val CLASSES = 7
    const val LOCAL_SPEAKERS = 3

    /** Bit k set = window-local speaker k is active. Index = class. */
    private val CLASS_SPEAKERS = byteArrayOf(0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110)

    /**
     * Most likely class per frame, as a speaker bit mask. [logProbabilities] is
     * row-major `[frames, CLASSES]`.
     */
    fun activeSpeakers(logProbabilities: FloatArray): ByteArray {
        require(logProbabilities.size % CLASSES == 0)
        val frames = logProbabilities.size / CLASSES
        return ByteArray(frames) { frame ->
            val row = frame * CLASSES
            var best = 0
            for (index in 1 until CLASSES) {
                if (logProbabilities[row + index] > logProbabilities[row + best]) best = index
            }
            CLASS_SPEAKERS[best]
        }
    }

    fun isActive(mask: Byte, localSpeaker: Int): Boolean = (mask.toInt() shr localSpeaker) and 1 == 1

    fun isOverlap(mask: Byte): Boolean = Integer.bitCount(mask.toInt()) > 1
}
