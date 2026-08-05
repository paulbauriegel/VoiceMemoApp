package com.example.voicememo.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

/**
 * Material 3 presentation root.
 *
 * This composable is deliberately state-driven and has no dependency on Navigation Compose.
 * The integrating Activity/ViewModel owns navigation and side effects by reducing
 * [VoiceMemoUiAction] values into a new [VoiceMemoAppUiState].
 */
@Composable
fun VoiceMemoApp(
    uiState: VoiceMemoAppUiState,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
    darkTheme: Boolean? = null,
    dynamicColor: Boolean = false,
) {
    val resolvedDarkTheme = darkTheme ?: uiState.appearanceMode.resolveDarkTheme(
        isSystemInDarkTheme(),
    )
    VoiceMemoTheme(
        darkTheme = resolvedDarkTheme,
        dynamicColor = dynamicColor,
    ) {
        Surface(
            modifier = modifier
                .fillMaxSize()
                .semantics {
                    // Expose stable Compose test tags to host-side UI Automator.
                    // This lets the non-destructive device test select controls
                    // by identity instead of screen coordinates.
                    testTagsAsResourceId = true
                }
                .testTag("voice_memo_app"),
            color = MaterialTheme.colorScheme.background,
        ) {
            VoiceMemoAdaptiveContent(
                uiState = uiState,
                onAction = onAction,
            )
        }

        uiState.pendingDelete?.let { request ->
            PermanentDeleteDialog(
                request = request,
                onDismiss = { onAction(VoiceMemoUiAction.DismissDelete) },
                onConfirm = { onAction(VoiceMemoUiAction.ConfirmDelete) },
            )
        }

        uiState.pendingShare?.let { request ->
            ShareChoiceSheet(
                request = request,
                onDismiss = { onAction(VoiceMemoUiAction.DismissShare) },
                onShare = { onAction(VoiceMemoUiAction.ShareAs(it)) },
            )
        }

        uiState.pendingProjectDelete?.let { project ->
            DeleteProjectDialog(
                project = project,
                onDismiss = { onAction(VoiceMemoUiAction.DismissDeleteProject) },
                onConfirm = { onAction(VoiceMemoUiAction.ConfirmDeleteProject) },
            )
        }
    }
}

@Composable
private fun VoiceMemoAdaptiveContent(
    uiState: VoiceMemoAppUiState,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val useListDetail = maxWidth >= 600.dp &&
            uiState.destination != VoiceMemoDestination.Settings

        if (useListDetail) {
            val listWidth = if (maxWidth >= 840.dp) 400.dp else 320.dp
            Row(modifier = Modifier.fillMaxSize()) {
                VoiceMemoLibraryScreen(
                    state = uiState,
                    onAction = onAction,
                    modifier = Modifier.width(listWidth),
                )
                VerticalDivider()
                Box(modifier = Modifier.weight(1f)) {
                    when (uiState.destination) {
                        VoiceMemoDestination.Recorder -> VoiceMemoRecorderScreen(
                            state = uiState.recorder,
                            onAction = onAction,
                            showBackButton = false,
                        )

                        VoiceMemoDestination.Detail -> {
                            val detail = uiState.detail
                            if (detail == null) {
                                EmptyDetailPane()
                            } else {
                                VoiceMemoDetailScreen(
                                    detail = detail,
                                    onAction = onAction,
                                    showBackButton = false,
                                )
                            }
                        }

                        VoiceMemoDestination.Library -> {
                            val detail = uiState.detail
                            if (detail == null || uiState.selectedMemoId == null) {
                                EmptyDetailPane()
                            } else {
                                VoiceMemoDetailScreen(
                                    detail = detail,
                                    onAction = onAction,
                                    showBackButton = false,
                                )
                            }
                        }

                        VoiceMemoDestination.Settings -> Unit
                    }
                }
            }
        } else {
            when (uiState.destination) {
                VoiceMemoDestination.Library -> VoiceMemoLibraryScreen(
                    state = uiState,
                    onAction = onAction,
                )

                VoiceMemoDestination.Recorder -> VoiceMemoRecorderScreen(
                    state = uiState.recorder,
                    onAction = onAction,
                )

                VoiceMemoDestination.Detail -> {
                    val detail = uiState.detail
                    if (detail == null) {
                        EmptyDetailPane()
                    } else {
                        VoiceMemoDetailScreen(
                            detail = detail,
                            onAction = onAction,
                        )
                    }
                }

                VoiceMemoDestination.Settings -> VoiceMemoLicensesScreen(
                    licenses = uiState.licenses,
                    isImportingAudio = uiState.isImportingAudio,
                    isExportingBackup = uiState.isExportingBackup,
                    isRestoringBackup = uiState.isRestoringBackup,
                    showRestoreBackupConfirmation = uiState.showRestoreBackupConfirmation,
                    appearanceMode = uiState.appearanceMode,
                    transcriptionTimestampsEnabled =
                        uiState.transcriptionTimestampsEnabled,
                    voiceActivityDetectionEnabled =
                        uiState.voiceActivityDetectionEnabled,
                    gpuAccelerationEnabled = uiState.gpuAccelerationEnabled,
                    transcribeWhileRecordingEnabled =
                        uiState.transcribeWhileRecordingEnabled,
                    transcriptionTimingEnabled = uiState.transcriptionTimingEnabled,
                    transcriptionModelId = uiState.transcriptionModelId,
                    onAction = onAction,
                )
            }
        }
    }
}

@Composable
private fun EmptyDetailPane() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier.weight(1f),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Rounded.GraphicEq,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = MaterialTheme.colorScheme.primary,
                )
                Text(
                    text = "Select a recording",
                    style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.padding(top = 16.dp),
                )
                Text(
                    text = "Audio and its local transcript appear here.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }
    }
}

@Preview(
    name = "Library - blue light",
    showBackground = true,
    widthDp = 412,
    heightDp = 892,
)
@Composable
private fun LibraryPreview() {
    VoiceMemoApp(
        uiState = VoiceMemoPreviewData.appState,
        onAction = {},
        darkTheme = false,
        dynamicColor = false,
    )
}

@Preview(
    name = "List detail - blue dark",
    showBackground = true,
    widthDp = 900,
    heightDp = 700,
)
@Composable
private fun ListDetailPreview() {
    VoiceMemoApp(
        uiState = VoiceMemoPreviewData.appState.copy(
            destination = VoiceMemoDestination.Detail,
        ),
        onAction = {},
        darkTheme = true,
        dynamicColor = false,
    )
}
