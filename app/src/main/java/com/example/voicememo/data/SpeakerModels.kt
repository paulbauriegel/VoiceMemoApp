package com.example.voicememo.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/** Someone whose voice can be recognised in recordings. Exactly one person is "me". */
@Entity(tableName = "speaker_persons")
class SpeakerPerson(
    @PrimaryKey val id: String,
    val name: String,
    val isSelf: Boolean,
    val createdAtMillis: Long,
)

/**
 * One voice sample of a person, reduced to a voiceprint. A person's profile is
 * the set of their samples; a recording is compared with every one of them, so
 * samples from different microphones or days all help.
 */
@Entity(
    tableName = "speaker_voice_samples",
    foreignKeys = [
        ForeignKey(
            entity = SpeakerPerson::class,
            parentColumns = ["id"],
            childColumns = ["personId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("personId")],
)
class SpeakerVoiceSample(
    @PrimaryKey val id: String,
    val personId: String,
    /** Voice model the vector came from; vectors of other models are ignored. */
    val modelId: String,
    /** Unit-length voiceprint, little-endian float32. */
    val vector: ByteArray,
    /** [ORIGIN_RECORDED] sample audio or a transcript [ORIGIN_SEGMENT] marked by the user. */
    val origin: String,
    /** Source recording and time range for [ORIGIN_SEGMENT] samples. */
    val memoId: String?,
    val startMillis: Long?,
    val endMillis: Long?,
    /** Microphone or route the sample was recorded through, when known. */
    val inputChannel: String?,
    /** Sample audio kept in app-private storage for [ORIGIN_RECORDED] samples. */
    val audioFileName: String?,
    val speechMillis: Long,
    val createdAtMillis: Long,
) {
    companion object {
        const val ORIGIN_RECORDED = "recorded"
        const val ORIGIN_SEGMENT = "segment"
    }
}

/**
 * A window-local speaker of one transcription run with its voiceprint. Kept so
 * a recording can be re-labelled after the voice profile changes without
 * running the models over the audio again.
 */
@Entity(
    tableName = "speaker_local_voiceprints",
    primaryKeys = ["jobId", "chunkIndex", "localSpeaker"],
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("memoId")],
)
class SpeakerLocalVoiceprint(
    val memoId: String,
    val jobId: String,
    val chunkIndex: Int,
    val localSpeaker: Int,
    val startSample: Int,
    /** Bit per segmentation frame, least significant bit first. */
    val activeFrames: ByteArray,
    /** Unit-length voiceprint, little-endian float32. */
    val vector: ByteArray,
)

/**
 * Speaker analysis of one transcription run. The row exists once every window
 * has been analysed; [timeline] is filled when a voice profile existed to
 * label it with.
 */
@Entity(
    tableName = "speaker_analyses",
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("memoId")],
)
class SpeakerAnalysis(
    @PrimaryKey val jobId: String,
    val memoId: String,
    val modelId: String,
    /** Length of the analysed audio in 16 kHz samples. */
    val sampleCount: Int,
    val voiceprintCount: Int,
    val analysedAtMillis: Long,
    /** Run-length encoded 10 ms labels (see SpeakerTimeline), or null without a profile. */
    val timeline: ByteArray?,
    /** Identifies the voice samples the timeline was made with. */
    val profileKey: String?,
    val labelledAtMillis: Long?,
)

/** The user's own label for one transcript segment; it wins over the timeline. */
@Entity(
    tableName = "speaker_segment_overrides",
    primaryKeys = ["jobId", "segmentId"],
    foreignKeys = [
        ForeignKey(
            entity = MemoRecord::class,
            parentColumns = ["id"],
            childColumns = ["memoId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("memoId")],
)
class SpeakerSegmentOverride(
    val memoId: String,
    val jobId: String,
    val segmentId: String,
    /** A SpeakerLabel name. */
    val label: String,
    val updatedAtMillis: Long,
)
