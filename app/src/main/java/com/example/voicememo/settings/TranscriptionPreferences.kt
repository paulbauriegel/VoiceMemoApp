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

    var gpuAccelerationEnabled: Boolean
        get() = preferences.getBoolean(
            KEY_GPU_ACCELERATION_ENABLED,
            BuildConfig.WHISPER_BACKEND_REQUESTED.equals("gpu", ignoreCase = true),
        )
        set(value) {
            preferences.edit().putBoolean(KEY_GPU_ACCELERATION_ENABLED, value).apply()
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
        const val KEY_TRANSCRIPTION_MODEL_ID = "transcription_model_id"
    }
}
