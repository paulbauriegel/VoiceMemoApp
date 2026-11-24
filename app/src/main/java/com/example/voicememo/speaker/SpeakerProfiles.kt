package com.example.voicememo.speaker

import com.example.voicememo.data.SpeakerPerson
import com.example.voicememo.data.SpeakerSegmentOverride
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.data.VoiceMemoDao
import com.example.voicememo.transcription.PcmResampler
import com.example.voicememo.transcription.VoiceActivityDetector
import com.example.voicememo.transcription.WavAudioReader
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * The people whose voice is known and their voice samples. Every change
 * re-labels the analysed recordings, which is cheap (see [SpeakerLabelIndex]).
 */
internal class SpeakerProfiles(
    private val dao: VoiceMemoDao,
    private val index: SpeakerLabelIndex,
    private val sampleDirectory: File,
    private val openEmbedder: () -> CamPlusPlusEmbedder,
    private val openVoiceActivityDetector: () -> VoiceActivityDetector?,
) {
    private val upgradeMutex = Mutex()

    sealed interface SampleResult {
        class Added(val sample: SpeakerVoiceSample) : SampleResult
        /** Not enough clear speech to make a voiceprint from. */
        data object TooLittleSpeech : SampleResult
    }

    /** The "me" person, created on first use. */
    suspend fun self(): SpeakerPerson =
        dao.getSpeakerPersons().firstOrNull { it.isSelf }
            ?: SpeakerPerson(
                id = UUID.randomUUID().toString(),
                name = "Me",
                isSelf = true,
                createdAtMillis = System.currentTimeMillis(),
            ).also { dao.upsertSpeakerPerson(it) }

    fun newSampleFile(): File {
        sampleDirectory.mkdirs()
        return File(sampleDirectory, "${UUID.randomUUID()}.wav")
    }

    /**
     * Adds a recorded voice sample ([audio] inside the sample directory). The
     * voiceprint is the average over 1.5-second stretches that contain speech.
     * The file is deleted when it cannot be used.
     */
    suspend fun addRecordedSample(personId: String, audio: File, inputChannel: String?): SampleResult =
        withContext(Dispatchers.Default) {
            val vectors = openEmbedder().use { embedder -> speechVoiceprints(audio, embedder) }
            if (vectors.size < MIN_SAMPLE_WINDOWS) {
                audio.delete()
                return@withContext SampleResult.TooLittleSpeech
            }
            val sample = SpeakerVoiceSample(
                id = UUID.randomUUID().toString(),
                personId = personId,
                modelId = CamPlusPlusEmbedder.MODEL_ID,
                vector = SpeakerCodec.bytes(SpeakerDecision.centroid(vectors)),
                origin = SpeakerVoiceSample.ORIGIN_RECORDED,
                memoId = null,
                startMillis = null,
                endMillis = null,
                inputChannel = inputChannel,
                audioFileName = audio.name,
                speechMillis = vectors.size * HOP_MILLIS,
                createdAtMillis = System.currentTimeMillis(),
            )
            dao.upsertSpeakerVoiceSample(sample)
            index.relabelAll()
            SampleResult.Added(sample)
        }

    /** Voiceprints of the 1.5-second stretches of [audio] that contain speech. */
    private fun speechVoiceprints(audio: File, embedder: CamPlusPlusEmbedder): List<FloatArray> {
        val samples = PcmResampler.toWhisperRate(WavAudioReader.read(audio))
        val vectors = ArrayList<FloatArray>()
        val detector = runCatching(openVoiceActivityDetector).getOrNull()
        try {
            var start = 0
            while (start + WINDOW_SAMPLES <= samples.size) {
                val window = samples.copyOfRange(start, start + WINDOW_SAMPLES)
                if (detector?.hasSpeech(window) { false } != false) {
                    embedder.embed(window)?.let(vectors::add)
                }
                start += HOP_SAMPLES
            }
        } finally {
            (detector as? AutoCloseable)?.close()
        }
        return vectors
    }

    /**
     * Recorded voice samples made with an earlier voice model get a voiceprint
     * from the current one, computed from their kept audio. Returns how many
     * were brought up to date. Samples taken from a transcript line follow
     * when their recording is analysed again (see [SpeakerLabelIndex.analyse]).
     */
    suspend fun upgradeRecordedSamples(): Int = withContext(Dispatchers.Default) {
        upgradeMutex.withLock {
            val stale = dao.getSpeakerVoiceSamples().filter { sample ->
                sample.origin == SpeakerVoiceSample.ORIGIN_RECORDED &&
                    sample.modelId != CamPlusPlusEmbedder.MODEL_ID
            }
            if (stale.isEmpty()) return@withLock 0
            var upgraded = 0
            openEmbedder().use { embedder ->
                for (sample in stale) {
                    val audio = sample.audioFileName
                        ?.let { File(sampleDirectory, it) }
                        ?.takeIf { it.parentFile == sampleDirectory && it.isFile }
                        ?: continue
                    val vectors = runCatching { speechVoiceprints(audio, embedder) }.getOrNull().orEmpty()
                    if (vectors.size < MIN_SAMPLE_WINDOWS) continue
                    dao.upsertSpeakerVoiceSample(sample.withVector(SpeakerDecision.centroid(vectors)))
                    upgraded++
                }
            }
            if (upgraded > 0) index.relabelAll()
            upgraded
        }
    }

    /**
     * "This is me" on a transcript segment: the stored voiceprints of whoever
     * the segmentation heard in that time range become a voice sample. Needs no
     * audio. The segment is also marked as that person's.
     */
    suspend fun addSegmentSample(
        personId: String,
        memoId: String,
        jobId: String,
        segmentId: String,
        startMillis: Long,
        endMillis: Long,
    ): SampleResult = withContext(Dispatchers.Default) {
        val vector = segmentVoiceprint(
            dao.getSpeakerLocalVoiceprints(jobId).map {
                LocalSpeakerVoiceprint(
                    chunkIndex = it.chunkIndex,
                    localSpeaker = it.localSpeaker,
                    startSample = it.startSample,
                    activeFrames = SpeakerCodec.bits(it.activeFrames, PyannoteSegmenter.FRAMES),
                    vector = SpeakerCodec.floats(it.vector),
                )
            },
            startMillis,
            endMillis,
        )
        setSegmentLabel(memoId, jobId, segmentId, SpeakerLabel.ME)
        if (vector == null) return@withContext SampleResult.TooLittleSpeech
        val sample = SpeakerVoiceSample(
            id = UUID.randomUUID().toString(),
            personId = personId,
            modelId = CamPlusPlusEmbedder.MODEL_ID,
            vector = SpeakerCodec.bytes(vector),
            origin = SpeakerVoiceSample.ORIGIN_SEGMENT,
            memoId = memoId,
            startMillis = startMillis,
            endMillis = endMillis,
            inputChannel = null,
            audioFileName = null,
            speechMillis = endMillis - startMillis,
            createdAtMillis = System.currentTimeMillis(),
        )
        dao.upsertSpeakerVoiceSample(sample)
        index.relabelAll()
        SampleResult.Added(sample)
    }

    suspend fun deleteSample(sampleId: String) {
        val sample = dao.getSpeakerVoiceSample(sampleId) ?: return
        dao.deleteSpeakerVoiceSample(sampleId)
        sample.audioFileName?.let { name ->
            File(sampleDirectory, name).takeIf { it.parentFile == sampleDirectory }?.delete()
        }
        index.relabelAll()
    }

    /** The user's own label for a segment; null removes it again. */
    suspend fun setSegmentLabel(
        memoId: String,
        jobId: String,
        segmentId: String,
        label: SpeakerLabel?,
    ) {
        if (label == null) {
            dao.deleteSpeakerSegmentOverride(jobId, segmentId)
        } else {
            dao.upsertSpeakerSegmentOverride(
                SpeakerSegmentOverride(memoId, jobId, segmentId, label.name, System.currentTimeMillis()),
            )
        }
    }

    companion object {
        private const val WINDOW_SAMPLES = 24_000
        private const val HOP_SAMPLES = 12_000
        private const val HOP_MILLIS = 750L
        /** About five seconds of speech. */
        private const val MIN_SAMPLE_WINDOWS = 6
        private const val MIN_FRAMES_IN_SEGMENT = 25

        /**
         * Average voiceprint of the window-local speakers that talk for at least
         * 0.4 s inside the time range; per window only the one talking most.
         */
        internal fun segmentVoiceprint(
            speakers: List<LocalSpeakerVoiceprint>,
            startMillis: Long,
            endMillis: Long,
        ): FloatArray? {
            val first = startMillis * PyannoteSegmenter.SAMPLE_RATE / 1_000
            val last = endMillis * PyannoteSegmenter.SAMPLE_RATE / 1_000
            val centre = PyannoteSegmenter.RECEPTIVE_FIELD_SAMPLES / 2
            val chosen = speakers
                .map { speaker ->
                    speaker to speaker.activeFrames.indices.count { frame ->
                        speaker.activeFrames[frame] &&
                            speaker.startSample + frame * PyannoteSegmenter.FRAME_SHIFT_SAMPLES + centre in first until last
                    }
                }
                .filter { it.second >= MIN_FRAMES_IN_SEGMENT }
                .groupBy { it.first.chunkIndex }
                .map { (_, candidates) -> candidates.maxBy { it.second }.first.vector }
            return if (chosen.isEmpty()) null else SpeakerDecision.centroid(chosen)
        }
    }
}
