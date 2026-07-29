package com.example.voicememo.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.voicememo.data.ProjectIcon
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
        composeRule.onNodeWithTag("library_search_header").assertIsDisplayed()
        composeRule.onNodeWithTag("library_search_field").assertIsDisplayed()
        composeRule.onNodeWithText("Search recordings and transcripts").assertIsDisplayed()
        composeRule.onNodeWithText("Project notes").assertIsDisplayed()
        composeRule.onNodeWithContentDescription("Start recording").performClick()

        assertEquals(VoiceMemoUiAction.StartRecording, lastAction)
    }

    @Test
    fun recordingsScreenMatchesFiltersCardsAndBottomNavigation() {
        val fourthMemo = VoiceMemoPreviewData.memoThree.copy(
            id = "memo-4",
            title = "Older recording",
        )
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    memos = listOf(fourthMemo) + VoiceMemoPreviewData.appState.memos,
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("recordings_filter_all").assertIsDisplayed()
        composeRule.onNodeWithTag("recordings_filter_unassigned").assertIsDisplayed()
        composeRule.onNodeWithTag("recordings_filter_starred").assertIsDisplayed()
        composeRule.onNodeWithTag("bottom_recordings").assertIsSelected()
        composeRule.onNodeWithTag("bottom_library").assertIsNotSelected()
        composeRule.onNodeWithTag("record_fab").assertIsDisplayed()
        composeRule.onNodeWithText("00:00").assertDoesNotExist()
        composeRule.onNodeWithTag("memo_card_memo-2").assertDoesNotExist()

        composeRule.onNodeWithTag("recordings_filter_starred").performClick()
        composeRule.onNodeWithTag("memo_card_memo-4").assertDoesNotExist()
        composeRule.onNodeWithTag("memo_card_memo-1").assertIsDisplayed()

        composeRule.onNodeWithTag("recordings_filter_all").performClick()
        composeRule.onNodeWithTag("memo_card_memo-4").assertIsDisplayed()
    }

    @Test
    fun libraryDestinationShowsProjectsAndFoldersTabs() {
        var lastAction: VoiceMemoUiAction? = null
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    libraryTab = LibraryTab.Projects,
                ),
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("library_search_field").assertIsDisplayed()
        composeRule.onNodeWithTag("library_collection_projects").assertIsDisplayed()
        composeRule.onNodeWithTag("library_collection_folders").assertIsDisplayed()
        composeRule.onNodeWithText("My Projects").assertIsDisplayed()
        composeRule.onNodeWithText("Work").assertIsDisplayed()
        composeRule.onNodeWithTag("bottom_recordings").assertIsNotSelected()
        composeRule.onNodeWithTag("bottom_library").assertIsSelected()

        composeRule.onNodeWithTag("bottom_recordings").performClick()
        assertEquals(
            VoiceMemoUiAction.LibraryTabChanged(LibraryTab.Recents),
            lastAction,
        )
    }

    @Test
    fun projectOverflowEditsNameIconAndArchivesWithoutDeleting() {
        var lastAction: VoiceMemoUiAction? = null
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    libraryTab = LibraryTab.Projects,
                    activeProjectId = "project-work",
                    activeProjectName = "Work",
                    activeProjectIcon = ProjectIcon.WORK,
                ),
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("project_edit_action").performClick()
        composeRule.onNodeWithTag("project_editor").assertIsDisplayed()
        composeRule.onNodeWithTag("library_bottom_bar").assertDoesNotExist()
        composeRule.onNodeWithTag("project_name_editor").performTextClearance()
        composeRule.onNodeWithTag("project_name_editor").performTextInput("Work notes")
        composeRule.onNodeWithTag("project_icon_lightbulb").performClick()
        composeRule.onNodeWithTag("save_project_action").performClick()

        assertEquals(
            VoiceMemoUiAction.UpdateProject(
                projectId = "project-work",
                name = "Work notes",
                icon = ProjectIcon.LIGHTBULB,
            ),
            lastAction,
        )

        composeRule.onNodeWithTag("project_edit_action").performClick()
        composeRule.onNodeWithTag("archive_project_action").performClick()
        assertEquals(
            VoiceMemoUiAction.SetProjectArchived(
                projectId = "project-work",
                archived = true,
            ),
            lastAction,
        )
    }

    @Test
    fun archivedProjectsCanBeExpandedAndOpened() {
        val archived = VoiceMemoPreviewData.projects.first().copy(isArchived = true)
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    libraryTab = LibraryTab.Projects,
                    projects = listOf(archived),
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("archived_projects_toggle").performClick()
        composeRule.onNodeWithTag("project_card_${archived.id}").assertIsDisplayed()
        composeRule.onNodeWithText("Archived").assertIsDisplayed()
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
            .performClick()

        assertEquals(VoiceMemoUiAction.CancelTranscription, lastAction)
    }

    @Test
    fun metadataGenerationProgressAppearsOnlyInAudioView() {
        var selectedSection by mutableStateOf(DetailSection.Audio)
        val detail = VoiceMemoPreviewData.detail.copy(
            isGeneratingMetadata = true,
        )
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = detail.copy(selectedSection = selectedSection),
                ),
                onAction = { action ->
                    if (action is VoiceMemoUiAction.DetailSectionChanged) {
                        selectedSection = action.section
                    }
                },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("metadata_generation_progress").assertIsDisplayed()
        composeRule.onNodeWithText("Creating title and summary").assertIsDisplayed()
        composeRule.onNodeWithTag("detail_section_transcript").performClick()
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("metadata_generation_progress").assertDoesNotExist()
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
            .performClick()

        assertEquals(VoiceMemoUiAction.ResetEditedTranscript, lastAction)
    }

    @Test
    fun transcriptRowsShowOnlyBeginningTimestamps() {
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = VoiceMemoPreviewData.detail,
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.onAllNodesWithText("00:28").assertCountEquals(1)
        composeRule.onNodeWithText("00:56").assertDoesNotExist()
    }

    @Test
    fun transcriptFocusPanelIncludesPlaybackAndSkipControls() {
        var lastAction: VoiceMemoUiAction? = null
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = VoiceMemoPreviewData.detail,
                ),
                onAction = { lastAction = it },
                dynamicColor = false,
            )
        }

        composeRule.onNodeWithTag("transcript_focus_panel").assertIsDisplayed()
        composeRule.onNodeWithContentDescription("Currently playing").assertIsDisplayed()

        composeRule.onNodeWithContentDescription("Skip back 5 seconds").performClick()
        assertEquals(VoiceMemoUiAction.DetailSeekBy(-5_000L), lastAction)

        composeRule.onNodeWithContentDescription("Skip forward 10 seconds").performClick()
        assertEquals(VoiceMemoUiAction.DetailSeekBy(10_000L), lastAction)
    }

    @Test
    fun transcriptAutomaticallyScrollsToTheActiveSegment() {
        val segments = (0 until 12).map { index ->
            val startMillis = index * 10_000L
            TranscriptSegmentUi(
                id = "segment-$index",
                startMillis = startMillis,
                endMillis = startMillis + 10_000L,
                startLabel = "segment $index",
                endLabel = "",
                text = "Transcript line $index",
            )
        }
        var playbackPositionMillis by mutableStateOf(0L)

        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = VoiceMemoPreviewData.detail.copy(
                        durationMillis = 120_000L,
                        playbackPositionMillis = playbackPositionMillis,
                        playbackFraction = playbackPositionMillis / 120_000f,
                        transcript = VoiceMemoPreviewData.detail.transcript.copy(
                            segments = segments,
                        ),
                    ),
                ),
                onAction = {},
                dynamicColor = false,
            )
        }

        composeRule.runOnIdle {
            playbackPositionMillis = 110_000L
        }

        composeRule.waitForIdle()
        composeRule.onNodeWithText("Transcript line 11").assertIsDisplayed()
    }
}
