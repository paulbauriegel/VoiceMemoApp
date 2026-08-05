@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.example.voicememo.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AudioFile
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Description
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.FileCopy
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlin.math.max

@Composable
fun VoiceMemoSearchField(
    query: String,
    onQueryChanged: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    TextField(
        value = query,
        onValueChange = onQueryChanged,
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp),
        placeholder = { Text("Search recordings and transcripts") },
        leadingIcon = {
            Icon(
                imageVector = Icons.Rounded.Search,
                contentDescription = null,
            )
        },
        trailingIcon = if (query.isNotEmpty()) {
            {
                IconButton(onClick = { onQueryChanged("") }) {
                    Icon(
                        imageVector = Icons.Rounded.Close,
                        contentDescription = "Clear search",
                    )
                }
            }
        } else {
            null
        },
        singleLine = true,
        shape = CircleShape,
        colors = TextFieldDefaults.colors(
            focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            disabledContainerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            focusedIndicatorColor = Color.Transparent,
            unfocusedIndicatorColor = Color.Transparent,
            disabledIndicatorColor = Color.Transparent,
        ),
    )
}

@Composable
fun VoiceMemoWaveform(
    amplitudes: List<Float>,
    progress: Float,
    description: String,
    modifier: Modifier = Modifier,
    activeColor: Color = MaterialTheme.colorScheme.primary,
    inactiveColor: Color = MaterialTheme.colorScheme.outlineVariant,
) {
    val displayAmplitudes = remember(amplitudes) {
        if (amplitudes.isEmpty()) {
            listOf(
                .12f, .18f, .22f, .16f, .28f, .20f, .14f, .24f, .19f, .13f,
                .17f, .22f, .15f, .20f, .12f, .18f, .24f, .16f, .14f, .22f,
                .18f, .13f, .19f, .15f,
            )
        } else {
            normalizeWaveformForDisplay(
                amplitudes.sampledForDisplay(maxBars = 72),
            )
        }
    }
    val safeProgress = progress.coerceIn(0f, 1f)

    Canvas(
        modifier = modifier
            .clipToBounds()
            .semantics {
                contentDescription = description
            },
    ) {
        val gap = 3.dp.toPx()
        val minimumBarWidth = 2.dp.toPx()
        val renderedAmplitudes = displayAmplitudes.sampledForDisplay(
            maxBars = maximumWaveformBarCount(
                width = size.width,
                minimumBarWidth = minimumBarWidth,
                gap = gap,
            ),
        )
        val availableWidth = size.width - gap * (renderedAmplitudes.size - 1)
        val barWidth = max(minimumBarWidth, availableWidth / renderedAmplitudes.size)
        val usedWidth = barWidth * renderedAmplitudes.size +
            gap * (renderedAmplitudes.size - 1)
        val startX = max(0f, (size.width - usedWidth) / 2f)
        val minimumHeight = 4.dp.toPx()
        val radius = barWidth / 2f

        renderedAmplitudes.forEachIndexed { index, rawAmplitude ->
            val amplitude = rawAmplitude.coerceIn(0.04f, 1f)
            val barHeight = max(minimumHeight, size.height * amplitude)
            val x = startX + index * (barWidth + gap)
            val y = (size.height - barHeight) / 2f
            val fraction = if (renderedAmplitudes.size == 1) 1f else {
                index.toFloat() / renderedAmplitudes.lastIndex
            }
            drawRoundRect(
                color = if (fraction <= safeProgress) activeColor else inactiveColor,
                topLeft = Offset(x, y),
                size = Size(barWidth, barHeight),
                cornerRadius = CornerRadius(radius, radius),
            )
        }
    }
}

/**
 * Applies display-only gain so quiet recordings remain visually readable.
 *
 * The 95th percentile, rather than the loudest bar, controls gain so a click or
 * handling spike cannot flatten the rest of the waveform. Gain is capped to
 * avoid turning near-silence into a misleading full-height signal.
 */
internal fun normalizeWaveformForDisplay(amplitudes: List<Float>): List<Float> {
    if (amplitudes.isEmpty()) return emptyList()
    val sanitized = amplitudes.map { amplitude ->
        amplitude.takeIf(Float::isFinite)?.coerceIn(0f, 1f) ?: 0f
    }
    val nonZero = sanitized.filter { it > 0f }.sorted()
    if (nonZero.isEmpty()) return sanitized
    val referenceIndex = ((nonZero.lastIndex * 95) / 100)
        .coerceIn(nonZero.indices)
    val reference = nonZero[referenceIndex]
    if (reference >= WAVEFORM_DISPLAY_TARGET || reference <= 0f) return sanitized
    val gain = (WAVEFORM_DISPLAY_TARGET / reference)
        .coerceAtMost(MAX_WAVEFORM_DISPLAY_GAIN)
    return sanitized.map { (it * gain).coerceIn(0f, 1f) }
}

internal fun maximumWaveformBarCount(
    width: Float,
    minimumBarWidth: Float,
    gap: Float,
): Int {
    if (width <= 0f || minimumBarWidth <= 0f) return 1
    return ((width + gap.coerceAtLeast(0f)) /
        (minimumBarWidth + gap.coerceAtLeast(0f)))
        .toInt()
        .coerceAtLeast(1)
}

private const val WAVEFORM_DISPLAY_TARGET = 0.72f
private const val MAX_WAVEFORM_DISPLAY_GAIN = 24f

private fun List<Float>.sampledForDisplay(maxBars: Int): List<Float> {
    if (size <= maxBars) return this
    val samplesPerBar = size.toFloat() / maxBars
    return List(maxBars) { outputIndex ->
        val start = (outputIndex * samplesPerBar).toInt().coerceIn(indices)
        val endExclusive = ((outputIndex + 1) * samplesPerBar)
            .toInt()
            .coerceIn(start + 1, size)
        subList(start, endExclusive).maxOrNull() ?: 0f
    }
}

@Composable
fun ModelAvailabilityCard(
    state: ModelAvailabilityUiState,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
) {
    if (
        state.phase == ModelAvailabilityPhase.Hidden ||
        state.phase == ModelAvailabilityPhase.Available
    ) {
        return
    }

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (state.phase) {
                ModelAvailabilityPhase.Error,
                ModelAvailabilityPhase.Unsupported,
                -> MaterialTheme.colorScheme.errorContainer

                else -> MaterialTheme.colorScheme.secondaryContainer
            },
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.Top,
        ) {
            when (state.phase) {
                ModelAvailabilityPhase.Checking ->
                    CircularProgressIndicator(modifier = Modifier.size(32.dp))

                ModelAvailabilityPhase.Error,
                ModelAvailabilityPhase.Unsupported,
                -> Icon(
                    imageVector = Icons.Rounded.ErrorOutline,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onErrorContainer,
                )

                else -> Unit
            }

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = state.headline,
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    text = state.supportingText,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                when {
                    state.progress != null -> LinearProgressIndicator(
                        progress = { state.progress.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                    )

                    state.phase == ModelAvailabilityPhase.Checking -> LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                if (state.phase == ModelAvailabilityPhase.Error) {
                    TextButton(onClick = onRetry) {
                        Text("Retry")
                    }
                }
            }
        }
    }
}

@Composable
fun TranscriptionProgress(
    phase: TranscriptionPhase,
    statusLabel: String,
    progress: Float?,
    timingLabel: String? = null,
    thermalStatusLabel: String? = null,
    realTimeFactorLabel: String? = null,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (phase == TranscriptionPhase.Complete) {
                Icon(
                    imageVector = Icons.Rounded.CheckCircle,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
            } else {
                CircularProgressIndicator(modifier = Modifier.size(20.dp))
            }
            Text(
                text = statusLabel,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        timingLabel?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.testTag("transcription_timing"),
            )
        }
        thermalStatusLabel?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.testTag("transcription_thermal_status"),
            )
        }
        realTimeFactorLabel?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.testTag("transcription_real_time_factor"),
            )
        }
        if (phase != TranscriptionPhase.Complete) {
            if (progress == null) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            } else {
                LinearProgressIndicator(
                    progress = { progress.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
fun PermanentDeleteDialog(
    request: PendingDeleteUi,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    val singular = request.count == 1
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                if (singular) {
                    "Permanently delete this recording?"
                } else {
                    "Permanently delete these recordings?"
                },
            )
        },
        text = {
            Text(
                if (singular) {
                    "Recording is not backed up. This will be permanently deleted. " +
                        "You can’t undo this action."
                } else {
                    "Recordings are not backed up. These will be permanently deleted. " +
                        "You can’t undo this action."
                },
            )
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                colors = ButtonDefaults.textButtonColors(
                    contentColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Text("Delete")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}

@Composable
fun DeleteProjectDialog(
    project: ProjectListItemUi,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Delete “${project.name}”?") },
        text = {
            Text(
                "The project will be removed, but its ${project.recordingCountLabel} " +
                    "will stay safely available in Recents.",
            )
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                colors = ButtonDefaults.textButtonColors(
                    contentColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Text("Delete project")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}

@Composable
fun ShareChoiceSheet(
    request: PendingShareUi,
    onDismiss: () -> Unit,
    onShare: (SharePayload) -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
        ) {
            Text(
                text = if (request.count == 1) "Share recording" else "Share recordings",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            )
            ShareChoice(
                label = "Audio",
                supportingText = "Share the original recording",
                icon = Icons.Rounded.AudioFile,
                enabled = true,
                onClick = { onShare(SharePayload.Audio) },
            )
            ShareChoice(
                label = "Transcript",
                supportingText = if (request.transcriptAvailableForAll) {
                    "Share transcript text"
                } else {
                    "Finish transcription before sharing text"
                },
                icon = Icons.Rounded.Description,
                enabled = request.transcriptAvailableForAll,
                onClick = { onShare(SharePayload.Transcript) },
            )
            ShareChoice(
                label = "Audio and transcript",
                supportingText = if (request.count == 1) {
                    "Share both files together"
                } else {
                    "Share all selected files together"
                },
                icon = Icons.Rounded.FileCopy,
                enabled = request.transcriptAvailableForAll,
                onClick = { onShare(SharePayload.AudioAndTranscript) },
            )
            Spacer(Modifier.height(8.dp))
            TextButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.End)
                    .padding(horizontal = 16.dp),
            ) {
                Text("Cancel")
            }
        }
    }
}

@Composable
private fun ShareChoice(
    label: String,
    supportingText: String,
    icon: ImageVector,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    ListItem(
        headlineContent = { Text(label) },
        supportingContent = { Text(supportingText) },
        leadingContent = {
            Icon(
                imageVector = icon,
                contentDescription = null,
            )
        },
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp),
        trailingContent = {
            Button(
                onClick = onClick,
                enabled = enabled,
            ) {
                Text("Share")
            }
        },
    )
}
