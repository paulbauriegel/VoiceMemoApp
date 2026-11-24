package com.example.voicememo.settings

import android.content.Context
import com.example.voicememo.BuildConfig
import com.example.voicememo.transcription.TranscriptionModel

class TranscriptionPreferences(
    context: Context,
) {
    private val preferences = context.applicationContext.getSharedPreferences(
        PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )

    var timestampsEnabled: Boolean
        get() = preferences.getBoolean(KEY_TIMESTAMPS_ENABLED, false)
        set(value) {
            preferences.edit().putBoolean(KEY_TIMESTAMPS_ENABLED, value).apply()
        }

    var voiceActivityDetectionEnabled: Boolean
        get() = preferences.getBoolean(KEY_VAD_ENABLED, true)
        set(value) {
            preferences.edit().putBoolean(KEY_VAD_ENABLED, value).apply()
        }

    /** "Me / others" speaker labels; off until the user turns them on. */
    var speakerLabelsEnabled: Boolean
        get() = preferences.getBoolean(KEY_SPEAKER_LABELS_ENABLED, false)
        set(value) {
            preferences.edit().putBoolean(KEY_SPEAKER_LABELS_ENABLED, value).apply()
        }

    /** Shows consecutive lines of one speaker as one longer line; stored lines stay as they are. */
    var joinSpeakerLinesEnabled: Boolean
        get() = preferences.getBoolean(KEY_JOIN_SPEAKER_LINES_ENABLED, false)
        set(value) {
            preferences.edit().putBoolean(KEY_JOIN_SPEAKER_LINES_ENABLED, value).apply()
        }

    var gpuAccelerationEnabled: Boolean
        get() = preferences.getBoolean(
            KEY_GPU_ACCELERATION_ENABLED,
            BuildConfig.WHISPER_BACKEND_REQUESTED.equals("gpu", ignoreCase = true),
        )
        set(value) {
            preferences.edit().putBoolean(KEY_GPU_ACCELERATION_ENABLED, value).apply()
        }

    var transcribeWhileRecordingEnabled: Boolean
        get() = preferences.getBoolean(KEY_TRANSCRIBE_WHILE_RECORDING_ENABLED, false)
        set(value) {
            preferences.edit().putBoolean(KEY_TRANSCRIBE_WHILE_RECORDING_ENABLED, value).apply()
        }

    var transcriptionTimingEnabled: Boolean
        get() = preferences.getBoolean(KEY_TRANSCRIPTION_TIMING_ENABLED, false)
        set(value) {
            preferences.edit().putBoolean(KEY_TRANSCRIPTION_TIMING_ENABLED, value).apply()
        }

    var transcriptionModelId: String
        get() = preferences.getString(
            KEY_TRANSCRIPTION_MODEL_ID,
            TranscriptionModel.DEFAULT.id,
        ) ?: TranscriptionModel.DEFAULT.id
        set(value) {
            preferences.edit().putString(
                KEY_TRANSCRIPTION_MODEL_ID,
                TranscriptionModel.fromId(value).id,
            ).apply()
        }

    private companion object {
        const val PREFERENCES_NAME = "transcription_settings"
        const val KEY_TIMESTAMPS_ENABLED = "whisper_timestamps_enabled"
        const val KEY_VAD_ENABLED = "voice_activity_detection_enabled"
        const val KEY_GPU_ACCELERATION_ENABLED = "gpu_acceleration_enabled"
        const val KEY_SPEAKER_LABELS_ENABLED = "speaker_labels_enabled"
        const val KEY_JOIN_SPEAKER_LINES_ENABLED = "join_speaker_lines_enabled"
        const val KEY_TRANSCRIBE_WHILE_RECORDING_ENABLED =
            "transcribe_while_recording_enabled"
        const val KEY_TRANSCRIPTION_TIMING_ENABLED = "transcription_timing_enabled"
        const val KEY_TRANSCRIPTION_MODEL_ID = "transcription_model_id"
    }
}
