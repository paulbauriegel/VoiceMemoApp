package com.example.voicememo.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VoiceMemoAppTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<ComposeTestHostActivity>()

    @Test
    fun libraryShowsSearchRecordingsAndRecordAction() {
        var lastAction: VoiceMemoUiAction? = null
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState,
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("library_screen").assertIsDisplayed()
        composeRule.onNodeWithText("Search recordings and transcripts").assertIsDisplayed()
        composeRule.onNodeWithText("Project notes").assertIsDisplayed()
        composeRule.onNodeWithContentDescription("Start recording").performClick()

        assertEquals(VoiceMemoUiAction.StartRecording, lastAction)
    }

    @Test
    fun librarySettingsActionOpensSettingsAndKeepsRecordAction() {
        var lastAction: VoiceMemoUiAction? = null
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState,
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithContentDescription("Settings").performClick()
        assertEquals(VoiceMemoUiAction.OpenSettings, lastAction)
        composeRule.onNodeWithContentDescription("Start recording").assertIsDisplayed()
    }

    @Test
    fun selectionModeHidesSettingsAction() {
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    selectedMemoIds = setOf(VoiceMemoPreviewData.memoOne.id),
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithContentDescription("Settings").assertDoesNotExist()
        composeRule.onNodeWithText("1 selected").assertIsDisplayed()
    }

    @Test
    fun searchScreenKeepsRankedPartialAndFuzzyResults() {
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    query = "projct",
                    memos = listOf(VoiceMemoPreviewData.memoOne),
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithText("1 result").assertIsDisplayed()
        composeRule.onNodeWithText("Project notes").assertIsDisplayed()
    }

    @Test
    fun permanentDeleteDialogUsesExplicitLocalOnlyCopy() {
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    pendingDelete = PendingDeleteUi(setOf("memo-1")),
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule
            .onNodeWithText("Permanently delete this recording?")
            .assertIsDisplayed()
        composeRule
            .onNodeWithText(
                "Recording is not backed up. This will be permanently deleted. " +
                    "You can’t undo this action.",
            )
            .assertIsDisplayed()
    }

    @Test
    fun runningTranscriptionExposesCancellation() {
        var lastAction: VoiceMemoUiAction? = null
        val detail = VoiceMemoPreviewData.detail.copy(
            transcript = VoiceMemoPreviewData.detail.transcript.copy(
                phase = TranscriptionPhase.Transcribing,
                statusLabel = "Transcribing 42%",
                progress = .42f,
            ),
        )
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = detail,
                ),
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule
            .onNodeWithText("Cancel transcription")
            .performScrollTo()
            .performClick()

        assertEquals(VoiceMemoUiAction.CancelTranscription, lastAction)
    }

    @Test
    fun editedTranscriptExposesResetAction() {
        var lastAction: VoiceMemoUiAction? = null
        val detail = VoiceMemoPreviewData.detail.copy(
            transcript = VoiceMemoPreviewData.detail.transcript.copy(isEdited = true),
        )
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = detail,
                ),
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule
            .onNodeWithText("Reset to generated transcript")
            .performScrollTo()
            .performClick()

        assertEquals(VoiceMemoUiAction.ResetEditedTranscript, lastAction)
    }
}
