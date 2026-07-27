package com.example.voicememo.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VoiceMemoLibraryHeaderTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<ComposeTestHostActivity>()

    @Test
    fun normalModeUsesSearchHeaderAndOneSettingsAction() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLibraryScreen(
                    state = VoiceMemoPreviewData.appState.copy(query = "project"),
                    onAction = actions::add,
                )
            }
        }

        composeRule.onNodeWithTag("library_search_header").assertIsDisplayed()
        composeRule.onNodeWithTag("library_search_field").assertIsDisplayed()
        composeRule.onNodeWithText("Voice memos").assertDoesNotExist()
        composeRule.onNodeWithContentDescription("Import audio").assertDoesNotExist()
        composeRule
            .onNodeWithContentDescription("About and open source licenses")
            .assertDoesNotExist()

        composeRule.onNodeWithContentDescription("Clear search").performClick()
        composeRule.onNodeWithContentDescription("Settings").performClick()

        assertEquals(
            listOf(
                VoiceMemoUiAction.SearchChanged(""),
                VoiceMemoUiAction.OpenSettings,
            ),
            actions,
        )
    }

    @Test
    fun projectModeUsesTitleAndExpandableSearch() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLibraryScreen(
                    state = VoiceMemoPreviewData.appState.copy(
                        libraryTab = LibraryTab.Projects,
                        activeProjectId = "project-work",
                        activeProjectName = "Work",
                    ),
                    onAction = actions::add,
                )
            }
        }

        composeRule.onNodeWithTag("project_header").assertIsDisplayed()
        composeRule.onNodeWithTag("project_title").assertIsDisplayed()
        composeRule.onNodeWithText("Work").assertIsDisplayed()
        composeRule.onNodeWithTag("project_search_field").assertDoesNotExist()
        composeRule.onNodeWithTag("library_search_header").assertDoesNotExist()
        composeRule.onNodeWithContentDescription("Settings").assertDoesNotExist()

        composeRule.onNodeWithContentDescription("Search in project").performClick()

        composeRule.onNodeWithTag("project_search_header").assertIsDisplayed()
        composeRule.onNodeWithTag("project_search_field").assertIsDisplayed()
        composeRule.onNodeWithTag("project_header").assertDoesNotExist()

        composeRule.onNodeWithContentDescription("Close project search").performClick()

        composeRule.onNodeWithTag("project_header").assertIsDisplayed()
        composeRule.onNodeWithTag("project_search_field").assertDoesNotExist()
        composeRule.onNodeWithContentDescription("Back to projects").performClick()
        assertEquals(listOf(VoiceMemoUiAction.CloseProject), actions)
    }

    @Test
    fun selectionModeReplacesSearchHeader() {
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLibraryScreen(
                    state = VoiceMemoPreviewData.appState.copy(
                        selectedMemoIds = setOf(VoiceMemoPreviewData.memoOne.id),
                    ),
                    onAction = {},
                )
            }
        }

        composeRule.onNodeWithText("1 selected").assertIsDisplayed()
        composeRule.onNodeWithTag("library_search_header").assertDoesNotExist()
        composeRule.onNodeWithContentDescription("Settings").assertDoesNotExist()
    }
}
