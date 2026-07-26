@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.example.voicememo.ui

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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Restore
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Translate
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.animation.animateColorAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

@Composable
fun VoiceMemoDetailScreen(
    detail: VoiceMemoDetailUi,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
    showBackButton: Boolean = true,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val largeText = LocalDensity.current.fontScale >= 1.3f

    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("detail_screen"),
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
                        text = detail.title,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                actions = {
                    IconButton(
                        onClick = {
                            onAction(VoiceMemoUiAction.RequestShare(setOf(detail.id)))
                        },
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Share,
                            contentDescription = "Share recording",
                        )
                    }
                    IconButton(
                        onClick = {
                            onAction(VoiceMemoUiAction.RequestDelete(setOf(detail.id)))
                        },
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Delete,
                            contentDescription = "Delete recording",
                        )
                    }
                },
            )
        },
    ) { innerPadding ->
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            val stackActions = largeText || maxWidth < 400.dp
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = spacing.sm, vertical = spacing.xs),
                verticalArrangement = Arrangement.spacedBy(spacing.md),
            ) {
                when (detail.selectedSection) {
                    DetailSection.Audio -> {
                        EditableTitle(
                            detail = detail,
                            largeText = stackActions,
                            onAction = onAction,
                        )
                        WaveformPlayer(
                            detail = detail,
                            onAction = onAction,
                        )
                        DetailSectionChooser(
                            selectedSection = detail.selectedSection,
                            onSelected = {
                                onAction(VoiceMemoUiAction.DetailSectionChanged(it))
                            },
                        )
                        AudioDetails(
                            detail = detail,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f),
                        )
                    }

                    DetailSection.Transcript -> {
                        DetailSectionChooser(
                            selectedSection = detail.selectedSection,
                            onSelected = {
                                onAction(VoiceMemoUiAction.DetailSectionChanged(it))
                            },
                        )
                        TranscriptDetails(
                            detail = detail,
                            transcript = detail.transcript,
                            largeText = stackActions,
                            onAction = onAction,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f),
                        )
                        TranscriptPlaybackControl(
                            detail = detail,
                            onAction = onAction,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EditableTitle(
    detail: VoiceMemoDetailUi,
    largeText: Boolean,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    if (detail.isEditingTitle) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedTextField(
                value = detail.titleDraft,
                onValueChange = {
                    onAction(VoiceMemoUiAction.TitleDraftChanged(it))
                },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Recording title") },
                singleLine = !largeText,
                maxLines = 2,
            )
            SaveCancelActions(
                stackVertically = largeText,
                saveEnabled = detail.titleDraft.isNotBlank(),
                onCancel = { onAction(VoiceMemoUiAction.CancelTitleEdit) },
                onSave = { onAction(VoiceMemoUiAction.SaveTitle) },
            )
        }
    } else {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = detail.title,
                    style = MaterialTheme.typography.headlineMedium,
                )
                Text(
                    text = detail.dateLabel,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            IconButton(
                onClick = { onAction(VoiceMemoUiAction.BeginTitleEdit) },
                modifier = Modifier.size(48.dp),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Edit,
                    contentDescription = "Rename recording",
                )
            }
        }
    }
}

@Composable
private fun WaveformPlayer(
    detail: VoiceMemoDetailUi,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainer,
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            VoiceMemoWaveform(
                amplitudes = detail.waveform,
                progress = detail.playbackFraction,
                description = "Audio waveform, ${detail.playbackPositionLabel} of ${detail.durationLabel}",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp),
            )
            Slider(
                value = detail.playbackFraction.coerceIn(0f, 1f),
                onValueChange = {
                    onAction(VoiceMemoUiAction.DetailSeekChanged(it))
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription =
                            "Playback position ${detail.playbackPositionLabel} of ${detail.durationLabel}"
                    },
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(
                    onClick = { onAction(VoiceMemoUiAction.ToggleDetailPlayback) },
                    modifier = Modifier.size(48.dp),
                ) {
                    Icon(
                        imageVector = if (detail.isPlaying) {
                            Icons.Rounded.Pause
                        } else {
                            Icons.Rounded.PlayArrow
                        },
                        contentDescription = if (detail.isPlaying) {
                            "Pause recording"
                        } else {
                            "Play recording"
                        },
                    )
                }
                Text(
                    text = detail.playbackPositionLabel,
                    style = MaterialTheme.typography.labelLarge,
                )
                HorizontalDivider(modifier = Modifier.weight(1f))
                Text(
                    text = detail.durationLabel,
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }
}

@Composable
private fun TranscriptPlaybackControl(
    detail: VoiceMemoDetailUi,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surfaceContainer,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            FilledTonalButton(
                onClick = { onAction(VoiceMemoUiAction.ToggleDetailPlayback) },
                modifier = Modifier.heightIn(min = 48.dp),
            ) {
                Icon(
                    imageVector = if (detail.isPlaying) {
                        Icons.Rounded.Pause
                    } else {
                        Icons.Rounded.PlayArrow
                    },
                    contentDescription = null,
                )
                Spacer(Modifier.size(8.dp))
                Text(if (detail.isPlaying) "Pause" else "Play audio")
            }
            Text(
                text = "${detail.playbackPositionLabel} / ${detail.durationLabel}",
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun DetailSectionChooser(
    selectedSection: DetailSection,
    onSelected: (DetailSection) -> Unit,
) {
    val sections = DetailSection.entries
    SingleChoiceSegmentedButtonRow(
        modifier = Modifier
            .fillMaxWidth()
            .selectableGroup(),
    ) {
        sections.forEachIndexed { index, section ->
            SegmentedButton(
                selected = selectedSection == section,
                onClick = { onSelected(section) },
                modifier = Modifier.testTag(
                    when (section) {
                        DetailSection.Audio -> "detail_section_audio"
                        DetailSection.Transcript -> "detail_section_transcript"
                    },
                ),
                shape = SegmentedButtonDefaults.itemShape(
                    index = index,
                    count = sections.size,
                ),
                label = {
                    Text(
                        when (section) {
                            DetailSection.Audio -> "Audio"
                            DetailSection.Transcript -> "Transcript"
                        },
                    )
                },
            )
        }
    }
}

@Composable
private fun AudioDetails(
    detail: VoiceMemoDetailUi,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Audio",
            style = MaterialTheme.typography.headlineSmall,
        )
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.surfaceContainerLow,
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                AudioMetadata(label = "Recorded", value = detail.dateLabel)
                AudioMetadata(label = "Duration", value = detail.durationLabel)
                AudioMetadata(label = "Storage", value = "On this device")
            }
        }
        Text(
            text = "Audio and transcription stay on this device.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun AudioMetadata(
    label: String,
    value: String,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
        )
    }
}

@Composable
private fun TranscriptDetails(
    detail: VoiceMemoDetailUi,
    transcript: TranscriptDetailUi,
    largeText: Boolean,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val busy = transcript.phase in setOf(
        TranscriptionPhase.Queued,
        TranscriptionPhase.Preparing,
        TranscriptionPhase.Transcribing,
        TranscriptionPhase.Finalizing,
    )

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        LanguageChooser(
            selectedCode = transcript.selectedLanguageCode,
            languages = transcript.languages,
            enabled = !busy && !transcript.isEditing,
            onSelected = {
                onAction(VoiceMemoUiAction.TranscriptLanguageChanged(it))
            },
        )

        when (transcript.phase) {
            TranscriptionPhase.NotRequested -> {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large,
                    color = MaterialTheme.colorScheme.secondaryContainer,
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Text(
                            text = "Transcribe locally",
                            style = MaterialTheme.typography.titleLarge,
                        )
                        Text(
                            text = "Whisper runs on this device. The recording is not uploaded.",
                            style = MaterialTheme.typography.bodyLarge,
                        )
                        Button(
                            onClick = {
                                onAction(VoiceMemoUiAction.RequestTranscription)
                            },
                            modifier = Modifier
                                .heightIn(min = 48.dp)
                                .testTag("transcribe_button"),
                        ) {
                            Text("Transcribe")
                        }
                    }
                }
            }

            TranscriptionPhase.Queued,
            TranscriptionPhase.Preparing,
            TranscriptionPhase.Transcribing,
            TranscriptionPhase.Finalizing,
            -> {
                TranscriptionProgress(
                    phase = transcript.phase,
                    statusLabel = transcript.statusLabel,
                    progress = transcript.progress,
                )
                OutlinedButton(
                    onClick = { onAction(VoiceMemoUiAction.CancelTranscription) },
                    modifier = Modifier.heightIn(min = 48.dp),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Close,
                        contentDescription = null,
                    )
                    Spacer(Modifier.size(8.dp))
                    Text("Cancel transcription")
                }
            }

            TranscriptionPhase.Error -> {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = MaterialTheme.colorScheme.onErrorContainer,
                    ),
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Text(
                            text = "Transcription couldn’t finish",
                            style = MaterialTheme.typography.titleLarge,
                        )
                        Text(
                            text = transcript.errorMessage
                                ?: "The audio is safe. Try transcription again.",
                            style = MaterialTheme.typography.bodyLarge,
                        )
                        FilledTonalButton(
                            onClick = {
                                onAction(VoiceMemoUiAction.RetryTranscription)
                            },
                        ) {
                            Icon(
                                imageVector = Icons.Rounded.Refresh,
                                contentDescription = null,
                            )
                            Spacer(Modifier.size(8.dp))
                            Text("Retry")
                        }
                    }
                }
            }

            TranscriptionPhase.Complete -> {
                TranscriptionProgress(
                    phase = transcript.phase,
                    statusLabel = transcript.statusLabel,
                    progress = 1f,
                )
                if (transcript.isEditing) {
                    OutlinedTextField(
                        value = transcript.draft,
                        onValueChange = {
                            onAction(VoiceMemoUiAction.TranscriptDraftChanged(it))
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .testTag("transcript_editor"),
                        label = { Text("Transcript") },
                    )
                    SaveCancelActions(
                        stackVertically = largeText,
                        saveEnabled = true,
                        onCancel = {
                            onAction(VoiceMemoUiAction.CancelTranscriptEdit)
                        },
                        onSave = {
                            onAction(VoiceMemoUiAction.SaveTranscript)
                        },
                    )
                } else {
                    TranscriptTextPanel(
                        transcript = transcript,
                        playbackPositionMillis = detail.playbackPositionMillis,
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f),
                    )
                    if (largeText) {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            TranscriptEditButton(
                                modifier = Modifier.fillMaxWidth(),
                                onAction = onAction,
                            )
                            RetranscribeButton(
                                modifier = Modifier.fillMaxWidth(),
                                onAction = onAction,
                            )
                            if (transcript.isEdited) {
                                ResetTranscriptButton(
                                    modifier = Modifier.fillMaxWidth(),
                                    onAction = onAction,
                                )
                            }
                        }
                    } else {
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                TranscriptEditButton(
                                    modifier = Modifier.weight(1f),
                                    onAction = onAction,
                                )
                                RetranscribeButton(
                                    modifier = Modifier.weight(1f),
                                    onAction = onAction,
                                )
                            }
                            if (transcript.isEdited) {
                                ResetTranscriptButton(
                                    modifier = Modifier.align(Alignment.End),
                                    onAction = onAction,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TranscriptTextPanel(
    transcript: TranscriptDetailUi,
    playbackPositionMillis: Long,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier.testTag("transcript_text"),
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surfaceContainerLow,
    ) {
        val segments = transcript.segments
        if (segments.isEmpty()) {
            SelectionContainer {
                Text(
                    text = when {
                        transcript.text.isNotEmpty() -> transcript.text
                        transcript.isEdited -> "Transcript cleared."
                        else -> "No speech detected."
                    },
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(20.dp),
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    horizontal = 12.dp,
                    vertical = 12.dp,
                ),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(
                    items = segments,
                    key = { it.id },
                ) { segment ->
                    val active = playbackPositionMillis >= segment.startMillis &&
                        playbackPositionMillis < segment.endMillis
                    TranscriptSegmentRow(
                        segment = segment,
                        active = active,
                    )
                }
            }
        }
    }
}

@Composable
private fun TranscriptSegmentRow(
    segment: TranscriptSegmentUi,
    active: Boolean,
) {
    val containerColor by animateColorAsState(
        targetValue = if (active) {
            MaterialTheme.colorScheme.primaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainerLow
        },
        label = "segmentContainerColor",
    )
    val contentColor = if (active) {
        MaterialTheme.colorScheme.onPrimaryContainer
    } else {
        MaterialTheme.colorScheme.onSurface
    }
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = containerColor,
        contentColor = contentColor,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Text(
                text = segment.startLabel,
                style = MaterialTheme.typography.labelLarge,
                color = if (active) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
                modifier = Modifier.width(56.dp),
            )
            SelectionContainer(
                modifier = Modifier.weight(1f),
            ) {
                Text(
                    text = segment.text,
                    style = if (active) {
                        MaterialTheme.typography.titleMedium
                    } else {
                        MaterialTheme.typography.bodyLarge
                    },
                    fontWeight = if (active) FontWeight.Bold else FontWeight.Normal,
                )
            }
            Text(
                text = segment.endLabel,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun LanguageChooser(
    selectedCode: String,
    languages: List<LanguageUi>,
    enabled: Boolean,
    onSelected: (String) -> Unit,
) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    val selected = languages.firstOrNull { it.code == selectedCode }
        ?: LanguageUi(selectedCode, selectedCode)

    Box {
        OutlinedButton(
            onClick = { expanded = true },
            enabled = enabled,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp),
        ) {
            Icon(
                imageVector = Icons.Rounded.Translate,
                contentDescription = null,
            )
            Spacer(Modifier.size(8.dp))
            Text(
                text = "Language: ${selected.label}",
                modifier = Modifier.weight(1f),
            )
            Icon(
                imageVector = Icons.Rounded.ExpandMore,
                contentDescription = null,
            )
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            languages.forEach { language ->
                DropdownMenuItem(
                    text = { Text(language.label) },
                    onClick = {
                        expanded = false
                        onSelected(language.code)
                    },
                )
            }
        }
    }
}

@Composable
private fun SaveCancelActions(
    stackVertically: Boolean,
    saveEnabled: Boolean,
    onCancel: () -> Unit,
    onSave: () -> Unit,
) {
    val cancelButton: @Composable (Modifier) -> Unit = { modifier ->
        OutlinedButton(
            onClick = onCancel,
            modifier = modifier.heightIn(min = 48.dp),
        ) {
            Icon(
                imageVector = Icons.Rounded.Close,
                contentDescription = null,
            )
            Spacer(Modifier.size(8.dp))
            Text("Cancel")
        }
    }
    val saveButton: @Composable (Modifier) -> Unit = { modifier ->
        Button(
            onClick = onSave,
            enabled = saveEnabled,
            modifier = modifier.heightIn(min = 48.dp),
        ) {
            Icon(
                imageVector = Icons.Rounded.Save,
                contentDescription = null,
            )
            Spacer(Modifier.size(8.dp))
            Text("Save")
        }
    }

    if (stackVertically) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            saveButton(Modifier.fillMaxWidth())
            cancelButton(Modifier.fillMaxWidth())
        }
    } else {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            cancelButton(Modifier.weight(1f))
            saveButton(Modifier.weight(1f))
        }
    }
}

@Composable
private fun TranscriptEditButton(
    modifier: Modifier,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    Button(
        onClick = { onAction(VoiceMemoUiAction.BeginTranscriptEdit) },
        modifier = modifier.heightIn(min = 48.dp),
    ) {
        Icon(
            imageVector = Icons.Rounded.Edit,
            contentDescription = null,
        )
        Spacer(Modifier.size(8.dp))
        Text("Edit")
    }
}

@Composable
private fun RetranscribeButton(
    modifier: Modifier,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    OutlinedButton(
        onClick = { onAction(VoiceMemoUiAction.RetryTranscription) },
        modifier = modifier
            .heightIn(min = 48.dp)
            .testTag("retranscribe_button"),
    ) {
        Icon(
            imageVector = Icons.Rounded.Refresh,
            contentDescription = null,
        )
        Spacer(Modifier.size(8.dp))
        Text("Transcribe again")
    }
}

@Composable
private fun ResetTranscriptButton(
    modifier: Modifier,
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    TextButton(
        onClick = { onAction(VoiceMemoUiAction.ResetEditedTranscript) },
        modifier = modifier.heightIn(min = 48.dp),
    ) {
        Icon(
            imageVector = Icons.Rounded.Restore,
            contentDescription = null,
        )
        Spacer(Modifier.size(8.dp))
        Text("Reset to generated transcript")
    }
}
