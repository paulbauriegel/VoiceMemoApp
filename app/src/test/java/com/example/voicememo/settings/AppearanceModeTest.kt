package com.example.voicememo.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AppearanceModeTest {
    @Test
    fun systemModeFollowsSystemTheme() {
        assertFalse(AppearanceMode.SYSTEM.resolveDarkTheme(systemInDarkTheme = false))
        assertTrue(AppearanceMode.SYSTEM.resolveDarkTheme(systemInDarkTheme = true))
    }

    @Test
    fun explicitModesOverrideSystemTheme() {
        assertFalse(AppearanceMode.LIGHT.resolveDarkTheme(systemInDarkTheme = true))
        assertTrue(AppearanceMode.DARK.resolveDarkTheme(systemInDarkTheme = false))
    }

    @Test
    fun storedValuesAreParsedSafely() {
        assertEquals(AppearanceMode.DARK, AppearanceMode.fromStoredValue("dark"))
        assertEquals(AppearanceMode.LIGHT, AppearanceMode.fromStoredValue("LIGHT"))
        assertEquals(AppearanceMode.SYSTEM, AppearanceMode.fromStoredValue("unknown"))
        assertEquals(AppearanceMode.SYSTEM, AppearanceMode.fromStoredValue(null))
    }
}
