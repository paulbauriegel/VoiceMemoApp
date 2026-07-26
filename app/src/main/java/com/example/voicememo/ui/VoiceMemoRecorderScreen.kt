@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.example.voicememo.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@Composable
fun VoiceMemoRecorderScreen(
    state: RecorderUiState,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
    showBackButton: Boolean = true,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val largeText = LocalDensity.current.fontScale >= 1.3f
    val statusLabel = when (state.phase) {
        RecorderPhase.Ready -> "Ready to record"
        RecorderPhase.Recording -> "Recording"
        RecorderPhase.Paused -> "Paused"
        RecorderPhase.Stopping -> "Saving recording"
    }

    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("recorder_screen"),
        topBar = {
            TopAppBar(
                navigationIcon = {
                    if (showBackButton) {
                        IconButton(onClick = { onAction(VoiceMemoUiAction.NavigateBack) }) {
                            Icon(
                                imageVector = Icons.Rounded.ArrowBack,
                                contentDescription = "Back to recordings",
                            )
                        }
                    }
                },
                title = {
                    Text(
                        text = state.title,
                        maxLines = if (largeText) 2 else 1,
                    )
                },
            )
        },
    ) { innerPadding ->
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            val stackControls = largeText || maxWidth < 400.dp
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = spacing.sm, vertical = spacing.xs),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(spacing.md),
            ) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(
                            min = if (largeText) 184.dp else 248.dp,
                            max = if (largeText) 248.dp else 400.dp,
                        ),
                    shape = MaterialTheme.shapes.extraLarge,
                    color = MaterialTheme.colorScheme.surfaceContainer,
                ) {
                    VoiceMemoWaveform(
                        amplitudes = state.waveform,
                        progress = if (state.phase == RecorderPhase.Ready) 0f else 1f,
                        description = "Live recording waveform",
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = spacing.md, vertical = 40.dp),
                        activeColor = MaterialTheme.colorScheme.primary,
                    )
                }

                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(
                        modifier = Modifier
                            .size(14.dp)
                            .background(
                                color = when (state.phase) {
                                    RecorderPhase.Recording -> MaterialTheme.colorScheme.error
                                    RecorderPhase.Paused -> MaterialTheme.colorScheme.tertiary
                                    else -> MaterialTheme.colorScheme.outline
                                },
                                shape = CircleShape,
                            ),
                    )
                    Text(
                        text = statusLabel,
                        style = MaterialTheme.typography.titleMedium,
                    )
                }

                Text(
                    text = state.elapsedLabel,
                    style = if (largeText) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.displayLarge
                    },
                    textAlign = TextAlign.Center,
                    modifier = Modifier.semantics {
                        contentDescription = "Elapsed time ${state.elapsedLabel}"
                    },
                )

                state.errorMessage?.let { message ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer,
                            contentColor = MaterialTheme.colorScheme.onErrorContainer,
                        ),
                    ) {
                        Text(
                            text = message,
                            style = MaterialTheme.typography.bodyLarge,
                            modifier = Modifier.padding(16.dp),
                        )
                    }
                }

                RecorderControls(
                    phase = state.phase,
                    stackVertically = stackControls,
                    onAction = onAction,
                )
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

@Composable
private fun RecorderControls(
    phase: RecorderPhase,
    stackVertically: Boolean,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    val pauseOrResume: @Composable (Modifier) -> Unit = { buttonModifier ->
        FilledTonalButton(
            onClick = {
                onAction(
                    if (phase == RecorderPhase.Paused) {
                        VoiceMemoUiAction.ResumeRecording
                    } else {
                        VoiceMemoUiAction.PauseRecording
                    },
                )
            },
            enabled = phase == RecorderPhase.Recording || phase == RecorderPhase.Paused,
            modifier = buttonModifier.heightIn(min = 64.dp),
        ) {
            Icon(
                imageVector = if (phase == RecorderPhase.Paused) {
                    Icons.Rounded.PlayArrow
                } else {
                    Icons.Rounded.Pause
                },
                contentDescription = null,
            )
            Spacer(Modifier.size(8.dp))
            Text(if (phase == RecorderPhase.Paused) "Resume" else "Pause")
        }
    }

    val stopOrStart: @Composable (Modifier) -> Unit = { buttonModifier ->
        if (phase == RecorderPhase.Ready) {
            Button(
                onClick = { onAction(VoiceMemoUiAction.StartRecording) },
                modifier = buttonModifier.heightIn(min = 64.dp),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Mic,
                    contentDescription = null,
                )
                Spacer(Modifier.size(8.dp))
                Text("Record")
            }
        } else {
            Button(
                onClick = { onAction(VoiceMemoUiAction.StopRecording) },
                enabled = phase != RecorderPhase.Stopping,
                modifier = buttonModifier.heightIn(min = 64.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer,
                ),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Stop,
                    contentDescription = null,
                )
                Spacer(Modifier.size(8.dp))
                Text(if (phase == RecorderPhase.Stopping) "Saving…" else "Stop")
            }
        }
    }

    if (stackVertically) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (phase != RecorderPhase.Ready) {
                pauseOrResume(Modifier.fillMaxWidth())
            }
            stopOrStart(Modifier.fillMaxWidth())
        }
    } else {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (phase != RecorderPhase.Ready) {
                pauseOrResume(Modifier.weight(1f))
            }
            stopOrStart(Modifier.weight(1f))
        }
    }
}
