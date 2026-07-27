package com.example.voicememo.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceMemoThemeTest {
    @Test
    fun lightSchemeUsesBlueIdentityWithAccessibleRolePairs() {
        assertEquals(Color(0xFF005AC1), BlueLightColorScheme.primary)
        assertEquals(Color(0xFFD8E2FF), BlueLightColorScheme.primaryContainer)
        assertEquals(Color(0xFFF9F9FF), BlueLightColorScheme.surface)

        assertContrast(BlueLightColorScheme.primary, BlueLightColorScheme.onPrimary)
        assertContrast(
            BlueLightColorScheme.primaryContainer,
            BlueLightColorScheme.onPrimaryContainer,
        )
        assertContrast(BlueLightColorScheme.surface, BlueLightColorScheme.onSurface)
    }

    @Test
    fun darkSchemeUsesBlueIdentityWithAccessibleRolePairs() {
        assertEquals(Color(0xFFADC6FF), BlueDarkColorScheme.primary)
        assertEquals(Color(0xFF004493), BlueDarkColorScheme.primaryContainer)
        assertEquals(Color(0xFF111318), BlueDarkColorScheme.surface)

        assertContrast(BlueDarkColorScheme.primary, BlueDarkColorScheme.onPrimary)
        assertContrast(
            BlueDarkColorScheme.primaryContainer,
            BlueDarkColorScheme.onPrimaryContainer,
        )
        assertContrast(BlueDarkColorScheme.surface, BlueDarkColorScheme.onSurface)
    }

    private fun assertContrast(background: Color, foreground: Color) {
        val lighter = maxOf(background.luminance(), foreground.luminance())
        val darker = minOf(background.luminance(), foreground.luminance())
        val ratio = (lighter + 0.05f) / (darker + 0.05f)
        assertTrue("Expected contrast >= 4.5, was $ratio", ratio >= 4.5f)
    }
}
