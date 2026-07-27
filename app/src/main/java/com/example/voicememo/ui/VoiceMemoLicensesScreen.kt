@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.example.voicememo.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.OpenInNew
import androidx.compose.material.icons.rounded.Archive
import androidx.compose.material.icons.rounded.AudioFile
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Restore
import androidx.compose.material.icons.rounded.Security
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp

@Composable
fun VoiceMemoLicensesScreen(
    licenses: List<LicenseUi>,
    isImportingAudio: Boolean,
    isExportingBackup: Boolean = false,
    isRestoringBackup: Boolean = false,
    showRestoreBackupConfirmation: Boolean = false,
    transcriptionTimestampsEnabled: Boolean = false,
    voiceActivityDetectionEnabled: Boolean = true,
    gpuAccelerationEnabled: Boolean = true,
    transcriptionModelId: String = "whisper-large-v3-turbo",
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val settingsListState = rememberLazyListState()
    var selectedLicenseKey by rememberSaveable { mutableStateOf<String?>(null) }
    val selectedLicense = licenses.firstOrNull {
        it.selectionKey == selectedLicenseKey
    }
    val isFileOperationInProgress =
        isImportingAudio || isExportingBackup || isRestoringBackup

    BackHandler(enabled = selectedLicense != null) {
        selectedLicenseKey = null
    }

    if (selectedLicense != null) {
        LicenseDetailScreen(
            license = selectedLicense,
            onBack = { selectedLicenseKey = null },
            onOpenSource = { url ->
                onAction(VoiceMemoUiAction.OpenLicenseSource(url))
            },
            modifier = modifier,
        )
        return
    }

    if (showRestoreBackupConfirmation) {
        RestoreBackupConfirmationDialog(onAction = onAction)
    }

    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("settings_screen"),
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = { onAction(VoiceMemoUiAction.NavigateBack) }) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                            contentDescription = "Back",
                        )
                    }
                },
                title = { Text("Settings") },
            )
        },
    ) { innerPadding ->
        LazyColumn(
            state = settingsListState,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(spacing.sm),
            verticalArrangement = Arrangement.spacedBy(spacing.sm),
        ) {
            item(key = "backup-heading") {
                SettingsSectionHeading("Backup")
            }

            item(key = "export-backup") {
                BackupSettingsCard(
                    title = if (isExportingBackup) "Exporting backup" else "Export backup",
                    supportingText = if (isExportingBackup) {
                        "Creating a local copy of your audio and transcripts"
                    } else {
                        "Save all audio and transcripts to a backup ZIP you choose"
                    },
                    icon = Icons.Rounded.Archive,
                    progressContentDescription = if (isExportingBackup) {
                        "Exporting backup"
                    } else {
                        null
                    },
                    enabled = !isFileOperationInProgress,
                    testTag = "settings_export_backup",
                    onClick = { onAction(VoiceMemoUiAction.ExportBackup) },
                )
            }

            item(key = "restore-backup") {
                BackupSettingsCard(
                    title = if (isRestoringBackup) "Restoring backup" else "Restore backup",
                    supportingText = if (isRestoringBackup) {
                        "Copying audio and transcripts into Voice Memo"
                    } else {
                        "Import audio and transcripts from a Voice Memo backup"
                    },
                    icon = Icons.Rounded.Restore,
                    progressContentDescription = if (isRestoringBackup) {
                        "Restoring backup"
                    } else {
                        null
                    },
                    enabled = !isFileOperationInProgress,
                    testTag = "settings_restore_backup",
                    onClick = { onAction(VoiceMemoUiAction.RequestRestoreBackup) },
                )
            }

            item(key = "backup-privacy-note") {
                Text(
                    text = "Backup files contain readable audio and transcript text. " +
                        "Store them somewhere you trust.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 8.dp),
                )
            }

            item(key = "transcription-heading") {
                SettingsSectionHeading("Transcription")
            }

            item(key = "transcription-model") {
                TranscriptionModelSettingsCard(
                    selectedModelId = transcriptionModelId,
                    onSelected = { modelId ->
                        onAction(VoiceMemoUiAction.TranscriptionModelChanged(modelId))
                    },
                )
            }

            item(key = "gpu-acceleration") {
                TranscriptionToggleSettingsCard(
                    headline = "Use GPU for transcription",
                    supportingText =
                        "On uses GPU acceleration. Off uses CPU/XNNPACK. " +
                            "GPU failures still fall back to CPU.",
                    enabled = gpuAccelerationEnabled,
                    testTag = "settings_gpu_acceleration",
                    onEnabledChange = { enabled ->
                        onAction(
                            VoiceMemoUiAction.GpuAccelerationChanged(enabled),
                        )
                    },
                )
            }

            item(key = "whisper-timestamps") {
                TranscriptionToggleSettingsCard(
                    headline = "Whisper timestamps",
                    supportingText =
                        "Add Whisper-generated segment times to new transcriptions. " +
                            "Existing transcripts change only when you transcribe them again.",
                    enabled = transcriptionTimestampsEnabled,
                    testTag = "settings_whisper_timestamps",
                    onEnabledChange = { enabled ->
                        onAction(
                            VoiceMemoUiAction.TranscriptionTimestampsChanged(enabled),
                        )
                    },
                )
            }

            item(key = "voice-activity-detection") {
                TranscriptionToggleSettingsCard(
                    headline = "Voice activity detection",
                    supportingText =
                        "Use the bundled Silero model to skip 28-second windows without " +
                            "speech. Turn this off to send every window to Whisper.",
                    enabled = voiceActivityDetectionEnabled,
                    testTag = "settings_voice_activity_detection",
                    onEnabledChange = { enabled ->
                        onAction(
                            VoiceMemoUiAction.VoiceActivityDetectionChanged(enabled),
                        )
                    },
                )
            }

            item(key = "audio-heading") {
                SettingsSectionHeading("Audio")
            }

            item(key = "import-audio") {
                ImportAudioSettingsCard(
                    isImportingAudio = isImportingAudio,
                    enabled = !isFileOperationInProgress,
                    onImportAudio = { onAction(VoiceMemoUiAction.ImportAudio) },
                )
            }

            item(key = "privacy-heading") {
                SettingsSectionHeading("Privacy")
            }

            item(key = "about") {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                    ),
                    shape = MaterialTheme.shapes.extraLarge,
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.Security,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                        )
                        Text(
                            text = "Private by design",
                            style = MaterialTheme.typography.headlineSmall,
                        )
                        Text(
                            text = "Voice memos and transcripts stay in Voice Memo's app storage " +
                                "unless you explicitly export or share them. Imported and restored " +
                                "files are copied into app storage. The bundled Whisper model " +
                                "performs transcription locally.",
                            style = MaterialTheme.typography.bodyLarge,
                        )
                    }
                }
            }

            item(key = "license-heading") {
                SettingsSectionHeading("Open source licenses")
            }

            if (licenses.isEmpty()) {
                item(key = "license-empty") {
                    Text(
                        text = "License information is not available.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            } else {
                item(key = "license-list") {
                    LicenseList(
                        licenses = licenses,
                        onLicenseSelected = { license ->
                            selectedLicenseKey = license.selectionKey
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun TranscriptionToggleSettingsCard(
    headline: String,
    supportingText: String,
    enabled: Boolean,
    testTag: String,
    onEnabledChange: (Boolean) -> Unit,
) {
    Card(
        onClick = { onEnabledChange(!enabled) },
        modifier = Modifier
            .fillMaxWidth()
            .testTag(testTag),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        ListItem(
            headlineContent = { Text(headline) },
            supportingContent = {
                Text(supportingText)
            },
            trailingContent = {
                Switch(
                    checked = enabled,
                    onCheckedChange = null,
                )
            },
            modifier = Modifier.heightIn(min = 72.dp),
            colors = ListItemDefaults.colors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer,
            ),
        )
    }
}

@Composable
private fun TranscriptionModelSettingsCard(
    selectedModelId: String,
    onSelected: (String) -> Unit,
) {
    val models = listOf(
        Triple(
            "whisper-large-v3-turbo",
            "Whisper Large v3 Turbo",
            "Existing baseline model; supports language selection and optional timestamps.",
        ),
        Triple(
            "parakeet-tdt-0.6b-v3",
            "Parakeet TDT 0.6B v3",
            "Faster 5-second multilingual model with automatic language detection.",
        ),
    )
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("settings_transcription_model"),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        Column {
            models.forEachIndexed { index, (id, name, description) ->
                ListItem(
                    headlineContent = { Text(name) },
                    supportingContent = { Text(description) },
                    trailingContent = {
                        RadioButton(
                            selected = selectedModelId == id,
                            onClick = null,
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 72.dp)
                        .clickable { onSelected(id) }
                        .testTag("settings_transcription_model_$id"),
                    colors = ListItemDefaults.colors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    ),
                )
                if (index != models.lastIndex) {
                    HorizontalDivider(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingsSectionHeading(
    title: String,
) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleLarge,
        color = MaterialTheme.colorScheme.onSurface,
        modifier = Modifier.padding(
            start = 8.dp,
            top = 8.dp,
            bottom = 4.dp,
        ),
    )
}

@Composable
private fun ImportAudioSettingsCard(
    isImportingAudio: Boolean,
    enabled: Boolean,
    onImportAudio: () -> Unit,
) {
    Card(
        onClick = onImportAudio,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .testTag("settings_import_audio"),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        ListItem(
            headlineContent = {
                Text(
                    text = if (isImportingAudio) "Importing audio" else "Import audio",
                )
            },
            supportingContent = {
                Text(
                    text = if (isImportingAudio) {
                        "Copying the selected recording into Voice Memo"
                    } else {
                        "Add a saved recording from this device"
                    },
                )
            },
            leadingContent = {
                Icon(
                    imageVector = Icons.Rounded.AudioFile,
                    contentDescription = null,
                )
            },
            trailingContent = {
                if (isImportingAudio) {
                    CircularProgressIndicator(
                        modifier = Modifier
                            .size(24.dp)
                            .semantics {
                                contentDescription = "Importing audio"
                            },
                        strokeWidth = 3.dp,
                    )
                } else {
                    Icon(
                        imageVector = Icons.Rounded.ChevronRight,
                        contentDescription = null,
                    )
                }
            },
            modifier = Modifier.heightIn(min = 72.dp),
            colors = ListItemDefaults.colors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer,
            ),
        )
    }
}

@Composable
private fun BackupSettingsCard(
    title: String,
    supportingText: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    progressContentDescription: String?,
    enabled: Boolean,
    testTag: String,
    onClick: () -> Unit,
) {
    Card(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .fillMaxWidth()
            .testTag(testTag),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        ListItem(
            headlineContent = { Text(title) },
            supportingContent = { Text(supportingText) },
            leadingContent = {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                )
            },
            trailingContent = {
                if (progressContentDescription != null) {
                    CircularProgressIndicator(
                        modifier = Modifier
                            .size(24.dp)
                            .semantics {
                                contentDescription = progressContentDescription
                            },
                        strokeWidth = 3.dp,
                    )
                } else {
                    Icon(
                        imageVector = Icons.Rounded.ChevronRight,
                        contentDescription = null,
                    )
                }
            },
            modifier = Modifier.heightIn(min = 72.dp),
            colors = ListItemDefaults.colors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer,
            ),
        )
    }
}

@Composable
private fun RestoreBackupConfirmationDialog(
    onAction: (VoiceMemoUiAction) -> Unit,
) {
    AlertDialog(
        onDismissRequest = { onAction(VoiceMemoUiAction.DismissRestoreBackup) },
        modifier = Modifier.testTag("restore_backup_confirmation"),
        title = { Text("Restore from backup?") },
        text = {
            Text(
                "Choose a Voice Memo backup to add its audio and transcripts to this device. " +
                    "Existing recordings are not changed or deleted. Restoring the same backup " +
                    "again may add duplicate copies. Voice Memo processes the backup on this device.",
            )
        },
        confirmButton = {
            TextButton(onClick = { onAction(VoiceMemoUiAction.ConfirmRestoreBackup) }) {
                Text("Choose backup")
            }
        },
        dismissButton = {
            TextButton(onClick = { onAction(VoiceMemoUiAction.DismissRestoreBackup) }) {
                Text("Cancel")
            }
        },
    )
}

@Composable
private fun LicenseList(
    licenses: List<LicenseUi>,
    onLicenseSelected: (LicenseUi) -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("settings_license_list"),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer,
        ),
        shape = MaterialTheme.shapes.large,
    ) {
        Column {
            licenses.forEachIndexed { index, license ->
                ListItem(
                    headlineContent = {
                        Text(
                            text = license.name,
                            style = MaterialTheme.typography.bodyLarge,
                        )
                    },
                    trailingContent = {
                        Icon(
                            imageVector = Icons.Rounded.ChevronRight,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 64.dp)
                        .clickable { onLicenseSelected(license) }
                        .testTag("settings_license_$index"),
                    colors = ListItemDefaults.colors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    ),
                )
                if (index != licenses.lastIndex) {
                    HorizontalDivider(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        color = MaterialTheme.colorScheme.outlineVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun LicenseDetailScreen(
    license: LicenseUi,
    onBack: () -> Unit,
    onOpenSource: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val spacing = LocalVoiceMemoSpacing.current
    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("license_detail_screen"),
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                            contentDescription = "Back to settings",
                        )
                    }
                },
                title = { Text("Open source license") },
            )
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(spacing.sm),
            verticalArrangement = Arrangement.spacedBy(spacing.sm),
        ) {
            item(key = "license-title") {
                Text(
                    text = license.name,
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(horizontal = 8.dp),
                )
            }
            item(key = "license-notice") {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer,
                    ),
                    shape = MaterialTheme.shapes.large,
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        license.version?.let {
                            Text(
                                text = "Version $it",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Text(
                            text = license.licenseName,
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.primary,
                        )
                        license.notice?.let {
                            Text(
                                text = it,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        license.sourceUrl?.let { sourceUrl ->
                            TextButton(onClick = { onOpenSource(sourceUrl) }) {
                                Text("View source")
                                Icon(
                                    imageVector = Icons.AutoMirrored.Rounded.OpenInNew,
                                    contentDescription = null,
                                    modifier = Modifier.padding(start = 8.dp),
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

private val LicenseUi.selectionKey: String
    get() = listOf(
        name,
        version.orEmpty(),
        licenseName,
    ).joinToString(separator = "|")
