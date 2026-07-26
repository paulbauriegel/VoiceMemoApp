package com.example.voicememo.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.Dp

private val WarmLightColorScheme = lightColorScheme(
    primary = Color(0xFF8F241E),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFFFDAD5),
    onPrimaryContainer = Color(0xFF3B0907),
    secondary = Color(0xFF775651),
    onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFFFDAD4),
    onSecondaryContainer = Color(0xFF2C1512),
    tertiary = Color(0xFF705C2E),
    onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFFFBDFA6),
    onTertiaryContainer = Color(0xFF251A00),
    error = Color(0xFFBA1A1A),
    onError = Color(0xFFFFFFFF),
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),
    background = Color(0xFFFFF8F6),
    onBackground = Color(0xFF231917),
    surface = Color(0xFFFFF8F6),
    onSurface = Color(0xFF231917),
    surfaceDim = Color(0xFFE8D6D2),
    surfaceBright = Color(0xFFFFF8F6),
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFFFF0ED),
    surfaceContainer = Color(0xFFFCE9E5),
    surfaceContainerHigh = Color(0xFFF6E3DF),
    surfaceContainerHighest = Color(0xFFF0DDD9),
    surfaceVariant = Color(0xFFF5DDD8),
    onSurfaceVariant = Color(0xFF534340),
    surfaceTint = Color(0xFF8F241E),
    outline = Color(0xFF85736F),
    outlineVariant = Color(0xFFD8C2BD),
    inverseSurface = Color(0xFF392E2B),
    inverseOnSurface = Color(0xFFFFEDEA),
    inversePrimary = Color(0xFFFFB4AA),
    scrim = Color(0xFF000000),
)

private val WarmDarkColorScheme = darkColorScheme(
    primary = Color(0xFFFFB4AA),
    onPrimary = Color(0xFF56100D),
    primaryContainer = Color(0xFF721814),
    onPrimaryContainer = Color(0xFFFFDAD5),
    secondary = Color(0xFFE7BDB6),
    onSecondary = Color(0xFF442925),
    secondaryContainer = Color(0xFF5D3F3A),
    onSecondaryContainer = Color(0xFFFFDAD4),
    tertiary = Color(0xFFDEC38C),
    onTertiary = Color(0xFF3E2E04),
    tertiaryContainer = Color(0xFF564419),
    onTertiaryContainer = Color(0xFFFBDFA6),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color(0xFFFFDAD6),
    background = Color(0xFF1C100C),
    onBackground = Color(0xFFF5DED9),
    surface = Color(0xFF1C100C),
    onSurface = Color(0xFFF5DED9),
    surfaceDim = Color(0xFF1C100C),
    surfaceBright = Color(0xFF46342F),
    surfaceContainerLowest = Color(0xFF150B08),
    surfaceContainerLow = Color(0xFF251713),
    surfaceContainer = Color(0xFF2C1B16),
    surfaceContainerHigh = Color(0xFF382620),
    surfaceContainerHighest = Color(0xFF44312B),
    surfaceVariant = Color(0xFF534340),
    onSurfaceVariant = Color(0xFFD8C2BD),
    surfaceTint = Color(0xFFFFB4AA),
    outline = Color(0xFFA18C87),
    outlineVariant = Color(0xFF534340),
    inverseSurface = Color(0xFFF5DED9),
    inverseOnSurface = Color(0xFF392E2B),
    inversePrimary = Color(0xFF8F241E),
    scrim = Color(0xFF000000),
)

private val VoiceMemoShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(28.dp),
)

@Immutable
data class VoiceMemoSpacing(
    val xxs: Dp = 4.dp,
    val xs: Dp = 8.dp,
    val sm: Dp = 16.dp,
    val md: Dp = 24.dp,
    val lg: Dp = 32.dp,
    val xl: Dp = 48.dp,
)

val LocalVoiceMemoSpacing = staticCompositionLocalOf { VoiceMemoSpacing() }

@Composable
fun VoiceMemoTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && darkTheme ->
            dynamicDarkColorScheme(context)

        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            dynamicLightColorScheme(context)

        darkTheme -> WarmDarkColorScheme
        else -> WarmLightColorScheme
    }

    CompositionLocalProvider(
        LocalVoiceMemoSpacing provides VoiceMemoSpacing(),
    ) {
        MaterialTheme(
            colorScheme = colorScheme,
            shapes = VoiceMemoShapes,
            content = content,
        )
    }
}
