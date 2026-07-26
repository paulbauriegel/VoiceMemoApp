@file:OptIn(
    androidx.compose.foundation.ExperimentalFoundationApi::class,
    androidx.compose.material3.ExperimentalMaterial3Api::class,
)

package com.example.voicememo.ui

import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.MoreVert
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.SelectAll
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LargeFloatingActionButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

@Composable
fun VoiceMemoLibraryScreen(
    state: VoiceMemoAppUiState,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val selectionMode = state.selectedMemoIds.isNotEmpty()
    var showSelectionActions by rememberSaveable { mutableStateOf(false) }
    val sections = remember(state.memos, state.query) {
        if (state.query.isBlank()) {
            state.memos
                .groupBy { it.monthLabel }
                .map { (month, memos) -> month to memos }
        } else {
            val resultLabel = if (state.memos.size == 1) "1 result" else "${state.memos.size} results"
            listOf(resultLabel to state.memos)
        }
    }

    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("library_screen"),
        topBar = {
            if (selectionMode) {
                SelectionTopBar(
                    selectedCount = state.selectedMemoIds.size,
                    onClose = { onAction(VoiceMemoUiAction.ClearSelection) },
                    onMore = { showSelectionActions = true },
                )
            } else {
                LibrarySearchTopBar(
                    query = state.query,
                    onQueryChanged = { onAction(VoiceMemoUiAction.SearchChanged(it)) },
                    onOpenSettings = { onAction(VoiceMemoUiAction.OpenSettings) },
                )
            }
        },
        floatingActionButton = {
            if (!selectionMode && !state.isImportingAudio) {
                LargeFloatingActionButton(
                    onClick = { onAction(VoiceMemoUiAction.StartRecording) },
                    modifier = Modifier.testTag("record_fab"),
                    shape = CircleShape,
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                    elevation = FloatingActionButtonDefaults.elevation(),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Mic,
                        contentDescription = "Start recording",
                        modifier = Modifier.size(36.dp),
                    )
                }
            }
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(
                start = spacing.sm,
                top = spacing.xs,
                end = spacing.sm,
                bottom = 128.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(key = "model-availability") {
                ModelAvailabilityCard(
                    state = state.modelAvailability,
                    onRetry = { onAction(VoiceMemoUiAction.RetryModelAvailabilityCheck) },
                )
            }

            if (state.memos.isEmpty()) {
                item(key = "empty") {
                    LibraryEmptyState(hasQuery = state.query.isNotBlank())
                }
            } else {
                sections.forEach { (sectionTitle, memos) ->
                    item(key = "section-$sectionTitle") {
                        Text(
                            text = sectionTitle,
                            style = if (state.query.isBlank()) {
                                MaterialTheme.typography.headlineSmall
                            } else {
                                MaterialTheme.typography.titleMedium
                            },
                            color = if (state.query.isBlank()) {
                                MaterialTheme.colorScheme.onSurface
                            } else {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            },
                            modifier = Modifier.padding(
                                start = 8.dp,
                                top = spacing.md,
                                bottom = spacing.xxs,
                            ),
                        )
                    }
                    items(
                        items = memos,
                        key = { it.id },
                    ) { memo ->
                        VoiceMemoCard(
                            memo = memo,
                            selected = memo.id in state.selectedMemoIds,
                            selectionMode = selectionMode,
                            onOpen = {
                                if (selectionMode) {
                                    onAction(VoiceMemoUiAction.ToggleMemoSelection(memo.id))
                                } else {
                                    onAction(VoiceMemoUiAction.OpenMemo(memo.id))
                                }
                            },
                            onLongPress = {
                                onAction(VoiceMemoUiAction.ToggleMemoSelection(memo.id))
                            },
                            onToggleSelection = {
                                onAction(VoiceMemoUiAction.ToggleMemoSelection(memo.id))
                            },
                            onTogglePlayback = {
                                onAction(VoiceMemoUiAction.ToggleMemoPlayback(memo.id))
                            },
                        )
                    }
                }
            }
        }
    }

    if (showSelectionActions) {
        SelectionActionsSheet(
            selectedCount = state.selectedMemoIds.size,
            onDismiss = { showSelectionActions = false },
            onDelete = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.RequestDelete(state.selectedMemoIds))
            },
            onShare = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.RequestShare(state.selectedMemoIds))
            },
            onSelectAll = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.SelectAll)
            },
        )
    }
}

@Composable
private fun LibrarySearchTopBar(
    query: String,
    onQueryChanged: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val spacing = LocalVoiceMemoSpacing.current
    Surface(
        color = MaterialTheme.colorScheme.surface,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(
                    start = spacing.sm,
                    top = spacing.xs,
                    end = spacing.xs,
                    bottom = spacing.xs,
                )
                .testTag("library_search_header"),
            horizontalArrangement = Arrangement.spacedBy(spacing.xs),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            VoiceMemoSearchField(
                query = query,
                onQueryChanged = onQueryChanged,
                modifier = Modifier
                    .weight(1f)
                    .testTag("library_search_field"),
            )
            IconButton(
                onClick = onOpenSettings,
                modifier = Modifier
                    .size(48.dp)
                    .testTag("settings_action"),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Settings,
                    contentDescription = "Settings",
                )
            }
        }
    }
}

@Composable
private fun SelectionTopBar(
    selectedCount: Int,
    onClose: () -> Unit,
    onMore: () -> Unit,
) {
    TopAppBar(
        navigationIcon = {
            IconButton(onClick = onClose) {
                Icon(
                    imageVector = Icons.Rounded.Close,
                    contentDescription = "Exit selection mode",
                )
            }
        },
        title = {
            Text(
                text = "$selectedCount selected",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        actions = {
            IconButton(onClick = onMore) {
                Icon(
                    imageVector = Icons.Rounded.MoreVert,
                    contentDescription = "Selection actions",
                )
            }
        },
    )
}

@Composable
private fun VoiceMemoCard(
    memo: VoiceMemoListItemUi,
    selected: Boolean,
    selectionMode: Boolean,
    onOpen: () -> Unit,
    onLongPress: () -> Unit,
    onToggleSelection: () -> Unit,
    onTogglePlayback: () -> Unit,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val selectedContentColor = MaterialTheme.colorScheme.onSecondaryContainer
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.extraLarge)
            .combinedClickable(
                onClick = onOpen,
                onLongClick = onLongPress,
                onLongClickLabel = "Select recording",
            )
            .semantics { role = Role.Button }
            .testTag("memo_card_${memo.id}"),
        shape = MaterialTheme.shapes.extraLarge,
        color = if (selected) {
            MaterialTheme.colorScheme.secondaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainer
        },
        contentColor = if (selected) {
            selectedContentColor
        } else {
            MaterialTheme.colorScheme.onSurface
        },
        tonalElevation = if (selected) 3.dp else 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(spacing.md),
            verticalArrangement = Arrangement.spacedBy(spacing.sm),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(spacing.sm),
                verticalAlignment = Alignment.Top,
            ) {
                Icon(
                    imageVector = Icons.Rounded.GraphicEq,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 3.dp),
                )
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(spacing.xxs),
                ) {
                    Text(
                        text = memo.title,
                        style = MaterialTheme.typography.titleLarge,
                    )
                    Text(
                        text = memo.dateLabel,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (selectionMode) {
                    Checkbox(
                        checked = selected,
                        onCheckedChange = { onToggleSelection() },
                        modifier = Modifier.size(48.dp),
                    )
                } else {
                    IconButton(
                        onClick = onTogglePlayback,
                        modifier = Modifier.size(48.dp),
                    ) {
                        Icon(
                            imageVector = if (memo.isPlaying) {
                                Icons.Rounded.Pause
                            } else {
                                Icons.Rounded.PlayArrow
                            },
                            contentDescription = if (memo.isPlaying) {
                                "Pause ${memo.title}"
                            } else {
                                "Play ${memo.title}"
                            },
                        )
                    }
                }
            }

            if (!memo.transcriptPreview.isNullOrBlank()) {
                Text(
                    text = memo.transcriptPreview,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }

            TranscriptionStatusRow(memo.transcription)

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(spacing.sm),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                VoiceMemoWaveform(
                    amplitudes = memo.waveform,
                    progress = memo.playbackFraction,
                    description = "Waveform for ${memo.title}",
                    activeColor = if (selected) {
                        selectedContentColor
                    } else {
                        MaterialTheme.colorScheme.primary
                    },
                    inactiveColor = if (selected) {
                        selectedContentColor.copy(alpha = 0.45f)
                    } else {
                        MaterialTheme.colorScheme.outlineVariant
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(36.dp),
                )
                Text(
                    text = memo.durationLabel,
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }
}

@Composable
private fun TranscriptionStatusRow(
    transcription: TranscriptionSummaryUi,
) {
    val inProgress = transcription.phase in setOf(
        TranscriptionPhase.Queued,
        TranscriptionPhase.Preparing,
        TranscriptionPhase.Transcribing,
        TranscriptionPhase.Finalizing,
    )

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            val icon = when (transcription.phase) {
                TranscriptionPhase.Complete -> Icons.Rounded.CheckCircle
                TranscriptionPhase.Error -> Icons.Rounded.ErrorOutline
                else -> null
            }
            if (icon != null) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                    tint = if (transcription.phase == TranscriptionPhase.Error) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.primary
                    },
                )
            }
            Text(
                text = transcription.statusLabel,
                style = MaterialTheme.typography.labelMedium,
                color = if (transcription.phase == TranscriptionPhase.Error) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }
        if (inProgress) {
            val progress = transcription.progress
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
private fun LibraryEmptyState(
    hasQuery: Boolean,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 64.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Icon(
            imageVector = if (hasQuery) Icons.Rounded.Info else Icons.Rounded.Mic,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Text(
            text = if (hasQuery) "No matching recordings" else "No recordings yet",
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = if (hasQuery) {
                "Try part of a word, another spelling, or fewer keywords."
            } else {
                "Tap Record to capture a new memo, or import saved audio from Settings."
            },
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun SelectionActionsSheet(
    selectedCount: Int,
    onDismiss: () -> Unit,
    onDelete: () -> Unit,
    onShare: () -> Unit,
    onSelectAll: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
        ) {
            Text(
                text = "$selectedCount selected",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            )
            SelectionAction(
                label = "Delete",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.Delete,
                        contentDescription = null,
                    )
                },
                onClick = onDelete,
            )
            SelectionAction(
                label = "Share",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.Share,
                        contentDescription = null,
                    )
                },
                onClick = onShare,
            )
            SelectionAction(
                label = "Select all",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.SelectAll,
                        contentDescription = null,
                    )
                },
                onClick = onSelectAll,
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
private fun SelectionAction(
    label: String,
    icon: @Composable () -> Unit,
    onClick: () -> Unit,
) {
    ListItem(
        headlineContent = { Text(label) },
        leadingContent = icon,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clickable(onClick = onClick),
    )
}
