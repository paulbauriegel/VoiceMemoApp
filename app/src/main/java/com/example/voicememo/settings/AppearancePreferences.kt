package com.example.voicememo.settings

import android.content.Context

enum class AppearanceMode {
    SYSTEM,
    LIGHT,
    DARK,
    ;

    fun resolveDarkTheme(systemInDarkTheme: Boolean): Boolean = when (this) {
        SYSTEM -> systemInDarkTheme
        LIGHT -> false
        DARK -> true
    }

    companion object {
        fun fromStoredValue(value: String?): AppearanceMode =
            entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: SYSTEM
    }
}

class AppearancePreferences(
    context: Context,
) {
    private val preferences = context.applicationContext.getSharedPreferences(
        PREFERENCES_NAME,
        Context.MODE_PRIVATE,
    )

    var mode: AppearanceMode
        get() = AppearanceMode.fromStoredValue(
            preferences.getString(KEY_APPEARANCE_MODE, null),
        )
        set(value) {
            preferences.edit().putString(KEY_APPEARANCE_MODE, value.name).apply()
        }

    private companion object {
        const val PREFERENCES_NAME = "appearance_settings"
        const val KEY_APPEARANCE_MODE = "appearance_mode"
    }
}
