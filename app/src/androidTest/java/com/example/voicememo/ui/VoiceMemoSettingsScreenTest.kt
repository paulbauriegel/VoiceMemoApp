package com.example.voicememo.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.voicememo.settings.AppearanceMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VoiceMemoSettingsScreenTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<ComposeTestHostActivity>()

    @Test
    fun appearanceRowsReflectSelectionAndDispatchModeChange() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    appearanceMode = AppearanceMode.SYSTEM,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithText("Appearance").assertIsDisplayed()
        composeRule.onNodeWithTag("settings_appearance_dark").performClick()
        assertEquals(
            VoiceMemoUiAction.AppearanceModeChanged(AppearanceMode.DARK),
            lastAction,
        )
    }

    @Test
    fun audioImportIsTheFirstSettingsSection() {
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    onAction = {},
                )
            }
        }

        val importTop = composeRule
            .onNodeWithTag("settings_import_audio")
            .assertIsDisplayed()
            .fetchSemanticsNode()
            .boundsInRoot
            .top
        val backupTop = composeRule
            .onNodeWithTag("settings_export_backup")
            .assertIsDisplayed()
            .fetchSemanticsNode()
            .boundsInRoot
            .top

        assertTrue("Audio import should appear before Backup", importTop < backupTop)
    }

    @Test
    fun openSourceLicensesAreCompactNamedRows() {
        val names = listOf(
            "OpenAI Whisper",
            "Google LiteRT",
            "Silero VAD",
            "Microsoft ONNX Runtime",
        )

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = names.map { name ->
                        LicenseUi(name = name, licenseName = "Test license")
                    },
                    isImportingAudio = false,
                    onAction = {},
                )
            }
        }

        names.forEach { name ->
            composeRule.onNodeWithText(name).performScrollTo().assertIsDisplayed()
        }
        composeRule.onNodeWithTag("settings_license_list").assertIsDisplayed()
    }

    @Test
    fun settingsPreservesPrivacyLicensesAndDispatchesImport() {
        var lastAction: VoiceMemoUiAction? = null
        val license = LicenseUi(
            name = "Example audio library",
            version = "1.2.3",
            licenseName = "Apache License 2.0",
            notice = "Complete required open-source notice.",
            sourceUrl = "https://example.com/source",
        )

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = listOf(license),
                    isImportingAudio = false,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithText("Settings").assertIsDisplayed()
        composeRule.onNodeWithText("Import audio").performScrollTo().performClick()
        assertEquals(VoiceMemoUiAction.ImportAudio, lastAction)
        composeRule.onNodeWithText("Private by design").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText(license.name).performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("settings_license_list").assertIsDisplayed()

        composeRule.onNodeWithText(license.name).performClick()
        composeRule.onNodeWithTag("license_detail_screen").assertIsDisplayed()
        composeRule.onNodeWithText("Version ${license.version}").assertIsDisplayed()
        composeRule.onNodeWithText(license.licenseName).assertIsDisplayed()
        composeRule.onNodeWithText(license.notice!!).assertIsDisplayed()
        composeRule.onNodeWithText("View source").performClick()
        assertEquals(VoiceMemoUiAction.OpenLicenseSource(license.sourceUrl!!), lastAction)

        composeRule.onNodeWithContentDescription("Back to settings").performClick()
        composeRule.onNodeWithTag("settings_screen").assertIsDisplayed()
        composeRule.onNodeWithText(license.name).assertIsDisplayed()
    }

    @Test
    fun whisperTimestampRowReflectsStateAndDispatchesToggle() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    transcriptionTimestampsEnabled = true,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("settings_whisper_timestamps")
            .performScrollTo()
            .performClick()
        assertEquals(
            VoiceMemoUiAction.TranscriptionTimestampsChanged(false),
            lastAction,
        )
        composeRule.onNodeWithText(
            "Add Whisper-generated segment times to new transcriptions. " +
                "Existing transcripts change only when you transcribe them again.",
        ).assertIsDisplayed()
    }

    @Test
    fun transcriptionTimingRowReflectsStateAndDispatchesToggle() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    transcriptionTimingEnabled = true,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("settings_transcription_timing")
            .performScrollTo()
            .performClick()
        assertEquals(
            VoiceMemoUiAction.TranscriptionTimingChanged(false),
            lastAction,
        )
    }

    @Test
    fun voiceActivityDetectionRowCanDisableVad() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    voiceActivityDetectionEnabled = true,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("settings_voice_activity_detection")
            .performScrollTo()
            .performClick()
        assertEquals(
            VoiceMemoUiAction.VoiceActivityDetectionChanged(false),
            lastAction,
        )
        composeRule.onNodeWithText(
            "Use the bundled Silero model to skip 28-second windows without " +
                "speech. Turn this off to send every window to Whisper.",
        ).assertIsDisplayed()
    }

    @Test
    fun gpuAccelerationRowCanSwitchToCpu() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    gpuAccelerationEnabled = true,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("settings_gpu_acceleration")
            .performScrollTo()
            .performClick()
        assertEquals(
            VoiceMemoUiAction.GpuAccelerationChanged(false),
            lastAction,
        )
        composeRule.onNodeWithText(
            "On uses GPU acceleration. Off uses CPU/XNNPACK. " +
                "GPU failures still fall back to CPU.",
        ).assertIsDisplayed()
    }

    @Test
    fun backupRowsExplainLocalExportAndRestoreAndDispatchActions() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithText("Backup").assertIsDisplayed()
        composeRule.onNodeWithText("Export backup").performClick()
        assertEquals(VoiceMemoUiAction.ExportBackup, lastAction)
        composeRule
            .onNodeWithText("Save all audio and transcripts to a backup ZIP you choose")
            .assertIsDisplayed()

        composeRule.onNodeWithText("Restore backup").performClick()
        assertEquals(VoiceMemoUiAction.RequestRestoreBackup, lastAction)
        composeRule
            .onNodeWithText("Import audio and transcripts from a Voice Memo backup")
            .assertIsDisplayed()
    }

    @Test
    fun restoreConfirmationExplainsLocalReadAndDispatchesConfirm() {
        var lastAction: VoiceMemoUiAction? = null

        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    showRestoreBackupConfirmation = true,
                    onAction = { lastAction = it },
                )
            }
        }

        composeRule.onNodeWithTag("restore_backup_confirmation").assertIsDisplayed()
        composeRule.onNodeWithText("Restore from backup?").assertIsDisplayed()
        composeRule
            .onNodeWithText(
                "Choose a Voice Memo backup to add its audio and transcripts to this device. " +
                    "Existing recordings are not changed or deleted. Restoring the same backup " +
                    "again may add duplicate copies. Voice Memo processes the backup on this device.",
            )
            .assertIsDisplayed()
        composeRule.onNodeWithText("Choose backup").performClick()
        assertEquals(VoiceMemoUiAction.ConfirmRestoreBackup, lastAction)
    }

    @Test
    fun importingAudioShowsProgressAndDisablesSettingsRow() {
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = true,
                    onAction = {},
                )
            }
        }

        composeRule.onNodeWithTag("settings_import_audio").assertIsNotEnabled()
        composeRule.onNodeWithContentDescription("Importing audio").assertIsDisplayed()
        composeRule
            .onNodeWithText("Copying the selected recording into Voice Memo")
            .assertIsDisplayed()
    }

    @Test
    fun backupProgressDisablesFileActionsAndHasAccessibleStatus() {
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    isExportingBackup = true,
                    onAction = {},
                )
            }
        }

        composeRule.onNodeWithTag("settings_export_backup").assertIsNotEnabled()
        composeRule.onNodeWithTag("settings_restore_backup").assertIsNotEnabled()
        composeRule.onNodeWithTag("settings_import_audio").assertIsNotEnabled()
        composeRule.onNodeWithContentDescription("Exporting backup").assertIsDisplayed()
        composeRule
            .onNodeWithText("Creating a local copy of your audio and transcripts")
            .assertIsDisplayed()
    }
}
