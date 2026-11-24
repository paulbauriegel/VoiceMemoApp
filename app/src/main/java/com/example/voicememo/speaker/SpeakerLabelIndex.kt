package com.example.voicememo.speaker

import com.example.voicememo.data.SpeakerAnalysis
import com.example.voicememo.data.SpeakerLocalVoiceprint
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.data.VoiceMemoDao
import com.example.voicememo.transcription.StreamingWavChunkSource
import com.example.voicememo.transcription.TranscriptionWindowConfig
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * "Me / others" labels for a finished transcription run.
 *
 * [analyse] is the expensive step: it runs the segmentation and voice models
 * over the audio once and stores a voiceprint per window-local speaker. It does
 * not depend on any voice profile. [relabel] is cheap: it compares the stored
 * voiceprints with the enrolled person's voice samples and stores the resulting
 * timeline, so it can run again whenever the profile changes.
 */
internal class SpeakerLabelIndex(
    private val dao: VoiceMemoDao,
    private val openSegmenter: () -> PyannoteSegmenter,
    private val openEmbedder: () -> CamPlusPlusEmbedder,
) {
    private val mutex = Mutex()
    private val mutableProgress = MutableStateFlow<Progress?>(null)

    /** The analysis running right now, if any; for progress display. */
    val progress: StateFlow<Progress?> = mutableProgress.asStateFlow()

    class Progress(val jobId: String, val fraction: Float)

    class Result(
        val windows: Int,
        val voiceprints: Int,
        val labelled: Boolean,
        val analysisMillis: Long,
    )

    /**
     * Analyses [audioFile] for transcription run [jobId] and labels it when a
     * voice profile exists. Returns null when [shouldPause] asked to stop first;
     * finished windows are kept and the next call continues after them.
     */
    suspend fun analyse(
        memoId: String,
        jobId: String,
        audioFile: File,
        shouldPause: () -> Boolean = { false },
        onProgress: suspend (window: Int, windows: Int) -> Unit = { _, _ -> },
    ): Result? = withContext(Dispatchers.Default) {
        mutex.withLock {
            val started = System.nanoTime()
            val existing = dao.getSpeakerAnalysis(jobId)
            if (existing != null && existing.modelId == CamPlusPlusEmbedder.MODEL_ID) {
                return@withLock Result(0, existing.voiceprintCount, relabelLocked(jobId), 0)
            }
            if (existing != null) {
                // Voiceprints of another voice model cannot be compared with the profile.
                dao.deleteSpeakerAnalysis(jobId)
                dao.deleteSpeakerLocalVoiceprints(jobId)
            }
            val stored = dao.getSpeakerLocalVoiceprints(jobId)
            // A window is stored in one transaction, so everything up to the last
            // stored one is complete. Silent windows after it are simply redone.
            val firstWindow = stored.maxOfOrNull { it.chunkIndex }?.plus(1) ?: 0
            var voiceprints = stored.size
            val windows: Int
            val sampleCount: Int
            try {
            StreamingWavChunkSource.open(audioFile, WINDOWS).use { source ->
                windows = source.chunkCount
                sampleCount = source.info.whisperSampleCount
                mutableProgress.value = Progress(jobId, firstWindow.toFloat() / windows.coerceAtLeast(1))
                if (firstWindow < windows) {
                    openSegmenter().use { segmenter ->
                        openEmbedder().use { embedder ->
                            val collector = SpeakerVoiceprintCollector(
                                segment = { segmenter.logProbabilities(it) },
                                embed = { embedder.embed(it) },
                            )
                            for (index in firstWindow until windows) {
                                if (shouldPause()) return@withLock null
                                val chunk = source.readChunk(index, shouldPause)
                                val found = collector.collect(index, chunk.window.startSample, chunk.samples)
                                if (found.isNotEmpty()) {
                                    dao.upsertSpeakerLocalVoiceprints(found.map { it.toRecord(memoId, jobId) })
                                    voiceprints += found.size
                                }
                                mutableProgress.value = Progress(jobId, (index + 1f) / windows)
                                onProgress(index + 1, windows)
                            }
                        }
                    }
                }
            }
            } finally {
                mutableProgress.value = null
            }
            upgradeSegmentSamples(memoId, jobId)
            dao.upsertSpeakerAnalysis(
                SpeakerAnalysis(
                    jobId = jobId,
                    memoId = memoId,
                    modelId = CamPlusPlusEmbedder.MODEL_ID,
                    sampleCount = sampleCount,
                    voiceprintCount = voiceprints,
                    analysedAtMillis = System.currentTimeMillis(),
                    timeline = null,
                    profileKey = null,
                    labelledAtMillis = null,
                ),
            )
            Result(
                windows = windows - firstWindow,
                voiceprints = voiceprints,
                labelled = relabelLocked(jobId),
                analysisMillis = (System.nanoTime() - started) / 1_000_000,
            )
        }
    }

    /** Labels an analysed run with the current voice profile. False without analysis or profile. */
    suspend fun relabel(jobId: String): Boolean = withContext(Dispatchers.Default) {
        mutex.withLock { relabelLocked(jobId) }
    }

    /** Re-labels every analysed run, after the voice profile changed. Returns how many got a timeline. */
    suspend fun relabelAll(): Int = withContext(Dispatchers.Default) {
        mutex.withLock { dao.getSpeakerAnalysisJobIds().count { relabelLocked(it) } }
    }

    /**
     * "This is me" samples taken from this recording with an earlier voice
     * model are recomputed from the voiceprints just stored for [jobId], so
     * they count for the profile again.
     */
    private suspend fun upgradeSegmentSamples(memoId: String, jobId: String) {
        val stale = dao.getSpeakerVoiceSamples().filter { sample ->
            sample.origin == SpeakerVoiceSample.ORIGIN_SEGMENT && sample.memoId == memoId &&
                sample.modelId != CamPlusPlusEmbedder.MODEL_ID
        }
        if (stale.isEmpty()) return
        val speakers = dao.getSpeakerLocalVoiceprints(jobId).map { it.toLocalSpeaker() }
        for (sample in stale) {
            val vector = SpeakerProfiles.segmentVoiceprint(
                speakers,
                sample.startMillis ?: continue,
                sample.endMillis ?: continue,
            ) ?: continue
            dao.upsertSpeakerVoiceSample(sample.withVector(vector))
        }
    }

    private suspend fun relabelLocked(jobId: String): Boolean {
        val analysis = dao.getSpeakerAnalysis(jobId) ?: return false
        if (analysis.modelId != CamPlusPlusEmbedder.MODEL_ID) return false
        val selfIds = dao.getSpeakerPersons().filter { it.isSelf }.map { it.id }.toSet()
        val samples = dao.getSpeakerVoiceSamples()
            .filter { it.personId in selfIds && it.modelId == CamPlusPlusEmbedder.MODEL_ID }
        if (samples.isEmpty()) {
            if (analysis.timeline != null) dao.upsertSpeakerAnalysis(analysis.with(null, null))
            return false
        }
        val profileKey = profileKey(samples.map { it.id })
        if (analysis.timeline != null && analysis.profileKey == profileKey) return true
        val speakers = dao.getSpeakerLocalVoiceprints(jobId).map { it.toLocalSpeaker() }
        val decider = SpeakerDecision.adapt(
            voiceprints = speakers.map { it.vector },
            profile = samples.map { SpeakerCodec.floats(it.vector) },
        )
        val timeline = SpeakerTimeline.build(speakers, analysis.sampleCount, decider::isMe)
        dao.upsertSpeakerAnalysis(analysis.with(timeline.encode(), profileKey))
        return true
    }

    private fun SpeakerAnalysis.with(timeline: ByteArray?, profileKey: String?) = SpeakerAnalysis(
        jobId = jobId,
        memoId = memoId,
        modelId = modelId,
        sampleCount = sampleCount,
        voiceprintCount = voiceprintCount,
        analysedAtMillis = analysedAtMillis,
        timeline = timeline,
        profileKey = profileKey,
        labelledAtMillis = timeline?.let { System.currentTimeMillis() },
    )

    private fun LocalSpeakerVoiceprint.toRecord(memoId: String, jobId: String) = SpeakerLocalVoiceprint(
        memoId = memoId,
        jobId = jobId,
        chunkIndex = chunkIndex,
        localSpeaker = localSpeaker,
        startSample = startSample,
        activeFrames = SpeakerCodec.bytes(activeFrames),
        vector = SpeakerCodec.bytes(vector),
    )

    private fun SpeakerLocalVoiceprint.toLocalSpeaker() = LocalSpeakerVoiceprint(
        chunkIndex = chunkIndex,
        localSpeaker = localSpeaker,
        startSample = startSample,
        activeFrames = SpeakerCodec.bits(activeFrames, PyannoteSegmenter.FRAMES),
        vector = SpeakerCodec.floats(vector),
    )

    companion object {
        private val WINDOWS = TranscriptionWindowConfig(
            payloadSamples = PyannoteSegmenter.WINDOW_SAMPLES,
            strideSamples = SpeakerVoiceprintCollector.HOP_SAMPLES,
        )

        /** Changes whenever the set of voice samples changes. */
        fun profileKey(sampleIds: List<String>): String {
            val digest = MessageDigest.getInstance("SHA-256")
            sampleIds.sorted().forEach { digest.update("$it\n".toByteArray()) }
            return digest.digest().take(8).joinToString("") { "%02x".format(it) }
        }
    }
}

/** The same voice sample with a voiceprint from the current voice model. */
internal fun SpeakerVoiceSample.withVector(vector: FloatArray) = SpeakerVoiceSample(
    id = id,
    personId = personId,
    modelId = CamPlusPlusEmbedder.MODEL_ID,
    vector = SpeakerCodec.bytes(vector),
    origin = origin,
    memoId = memoId,
    startMillis = startMillis,
    endMillis = endMillis,
    inputChannel = inputChannel,
    audioFileName = audioFileName,
    speechMillis = speechMillis,
    createdAtMillis = createdAtMillis,
)

/** Database encodings: little-endian float32 vectors and bit sets, least significant bit first. */
internal object SpeakerCodec {
    fun bytes(vector: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(vector.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(vector)
        return buffer.array()
    }

    fun floats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    fun bytes(bits: BooleanArray): ByteArray {
        val bytes = ByteArray((bits.size + 7) / 8)
        for (index in bits.indices) {
            if (bits[index]) bytes[index / 8] = (bytes[index / 8].toInt() or (1 shl (index % 8))).toByte()
        }
        return bytes
    }

    fun bits(bytes: ByteArray, count: Int): BooleanArray =
        BooleanArray(count) { index -> (bytes[index / 8].toInt() shr (index % 8)) and 1 == 1 }
}
