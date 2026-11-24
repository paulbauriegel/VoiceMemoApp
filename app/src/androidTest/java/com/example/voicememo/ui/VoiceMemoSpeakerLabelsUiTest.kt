package com.example.voicememo.ui

import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasScrollToNodeAction
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VoiceMemoSpeakerLabelsUiTest {
    @get:Rule
    val composeRule = createAndroidComposeRule<ComposeTestHostActivity>()

    private fun segments() = listOf(SpeakerUi.Me, SpeakerUi.Other, SpeakerUi.Mixed, null).mapIndexed { index, speaker ->
        TranscriptSegmentUi(
            id = "segment-$index",
            startMillis = index * 10_000L,
            endMillis = index * 10_000L + 10_000L,
            startLabel = "00:${index}0",
            endLabel = "",
            text = "Transcript line $index",
            speaker = speaker,
        )
    }

    /** The detail screen also resets its semantic search when it opens. */
    private fun List<VoiceMemoUiAction>.speakerActions() =
        filterNot { it is VoiceMemoUiAction.SearchTranscriptSemantically }

    private fun showDetail(
        status: SpeakerLabelStatus,
        progress: Float? = null,
        editingSegmentId: String? = null,
        segments: List<TranscriptSegmentUi> = segments(),
        joinSpeakerLines: Boolean = false,
        onAction: (VoiceMemoUiAction) -> Unit = {},
    ) {
        composeRule.setContent {
            VoiceMemoApp(
                uiState = VoiceMemoPreviewData.appState.copy(
                    destination = VoiceMemoDestination.Detail,
                    detail = VoiceMemoPreviewData.detail.copy(
                        durationMillis = 40_000L,
                        transcript = VoiceMemoPreviewData.detail.transcript.copy(
                            segments = segments,
                            speakerStatus = status,
                            joinSpeakerLines = joinSpeakerLines,
                            speakerProgress = progress,
                            editingSegmentId = editingSegmentId,
                            segmentDraft = "Transcript line 1",
                        ),
                    ),
                ),
                onAction = onAction,
                dynamicColor = false,
            )
        }
    }

    @Test
    fun labelledTranscriptShowsASpeakerIconBeforeEachTimestamp() {
        showDetail(SpeakerLabelStatus.Labelled)

        composeRule.onNodeWithTag("segment_speaker_Me", useUnmergedTree = true).assertIsDisplayed()
        composeRule.onNodeWithTag("segment_speaker_Other", useUnmergedTree = true).assertIsDisplayed()
        composeRule.onNodeWithTag("segment_speaker_Mixed", useUnmergedTree = true).assertIsDisplayed()
        composeRule.onNodeWithTag("segment_speaker_Unknown", useUnmergedTree = true).assertIsDisplayed()
        composeRule.onNodeWithTag("speaker_status_banner").assertDoesNotExist()
        // The legend names every icon.
        composeRule.onNodeWithTag("speaker_legend").assertIsDisplayed()
        listOf("You", "Others", "Both", "Unclear").forEach { label ->
            composeRule.onNodeWithText(label).assertIsDisplayed()
        }
    }

    @Test
    fun analysisShowsAProgressBarWithPercent() {
        showDetail(SpeakerLabelStatus.Analysing, progress = 0.42f)

        composeRule.onNodeWithTag("speaker_progress").assertIsDisplayed()
        composeRule.onNodeWithText("42%", substring = true).assertIsDisplayed()
        composeRule.onNodeWithTag("speaker_legend").assertDoesNotExist()
    }

    @Test
    fun speakerIconMenuMarksALineAsMe() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        showDetail(SpeakerLabelStatus.Labelled, onAction = actions::add)

        composeRule.onNodeWithTag("segment_speaker_Other", useUnmergedTree = true).performClick()
        composeRule.onNodeWithText("This is me").performClick()

        assertEquals(
            listOf<VoiceMemoUiAction>(VoiceMemoUiAction.SetSegmentSpeaker("segment-1", SpeakerUi.Me)),
            actions.speakerActions(),
        )
    }

    @Test
    fun lineEditorChangesTheSpeaker() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        showDetail(SpeakerLabelStatus.Labelled, editingSegmentId = "segment-1", onAction = actions::add)

        composeRule.onNodeWithTag("segment_editor_speaker").assertIsDisplayed()
        composeRule.onNodeWithText("Me").performClick()

        assertEquals(
            listOf<VoiceMemoUiAction>(VoiceMemoUiAction.SetSegmentSpeaker("segment-1", SpeakerUi.Me)),
            actions.speakerActions(),
        )
    }

    @Test
    fun lineEditorHasNoSpeakerChoiceWhenLabelsAreOff() {
        showDetail(SpeakerLabelStatus.Hidden, editingSegmentId = "segment-1")

        composeRule.onNodeWithTag("transcript_segment_editor").assertIsDisplayed()
        composeRule.onNodeWithTag("segment_editor_speaker").assertDoesNotExist()
    }

    @Test
    fun speakerIconsAreHiddenWhenLabelsAreOff() {
        showDetail(SpeakerLabelStatus.Hidden)

        composeRule.onNodeWithText("Transcript line 0").assertIsDisplayed()
        composeRule.onAllNodesWithTag("segment_speaker_Me", useUnmergedTree = true).assertCountEquals(0)
        composeRule.onNodeWithTag("speaker_status_banner").assertDoesNotExist()
    }

    @Test
    fun unanalysedRecordingOffersToIdentifySpeakers() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        showDetail(SpeakerLabelStatus.NotAnalysed, onAction = actions::add)

        composeRule.onNodeWithTag("speaker_status_banner").assertIsDisplayed()
        composeRule.onAllNodesWithTag("segment_speaker_Me", useUnmergedTree = true).assertCountEquals(0)
        composeRule.onNodeWithText("Identify").performClick()

        assertEquals(listOf<VoiceMemoUiAction>(VoiceMemoUiAction.IdentifySpeakers), actions.speakerActions())
    }

    /** Two lines of the user, then one of someone else. */
    private fun conversation() = listOf(SpeakerUi.Me, SpeakerUi.Me, SpeakerUi.Other).mapIndexed { index, speaker ->
        TranscriptSegmentUi(
            id = "segment-$index",
            startMillis = index * 3_000L,
            endMillis = index * 3_000L + 3_000L,
            startLabel = "00:0${index * 3}",
            endLabel = "",
            text = "Transcript line $index",
            speaker = speaker,
        )
    }

    @Test
    fun joinedLinesOfOneSpeakerShowAsOneLineAndChangeSpeakerTogether() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        showDetail(
            status = SpeakerLabelStatus.Labelled,
            segments = conversation(),
            joinSpeakerLines = true,
            onAction = actions::add,
        )

        composeRule.onNodeWithText("Transcript line 0 Transcript line 1").assertIsDisplayed()
        composeRule.onNodeWithText("Transcript line 2").assertIsDisplayed()
        composeRule.onAllNodesWithTag("segment_speaker_Me", useUnmergedTree = true).assertCountEquals(1)
        composeRule.onNodeWithTag("segment_speaker_Me", useUnmergedTree = true).performClick()
        composeRule.onNodeWithText("Someone else").performClick()

        assertEquals(
            listOf(VoiceMemoUiAction.SetSegmentSpeaker("segment-0", SpeakerUi.Other, listOf("segment-1"))),
            actions.speakerActions(),
        )
    }

    @Test
    fun linesStaySeparateWhenJoiningIsOff() {
        showDetail(status = SpeakerLabelStatus.Labelled, segments = conversation())

        composeRule.onNodeWithText("Transcript line 0").assertIsDisplayed()
        composeRule.onNodeWithText("Transcript line 1").assertIsDisplayed()
        composeRule.onAllNodesWithTag("segment_speaker_Me", useUnmergedTree = true).assertCountEquals(2)
    }

    @Test
    fun settingsOfferToJoinShortLinesWhenSpeakerLabelsAreOn() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    speakerLabelsAvailable = true,
                    speakerLabelsEnabled = true,
                    onAction = actions::add,
                )
            }
        }

        composeRule.onNode(hasScrollToNodeAction()).performScrollToNode(hasTestTag("settings_join_speaker_lines"))
        composeRule.onNodeWithTag("settings_join_speaker_lines").performClick()

        assertEquals(listOf<VoiceMemoUiAction>(VoiceMemoUiAction.JoinSpeakerLinesChanged(true)), actions)
    }

    @Test
    fun settingsListPeopleWithTheirSamplesAndRecordsANewOne() {
        val actions = mutableListOf<VoiceMemoUiAction>()
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(
                    licenses = emptyList(),
                    isImportingAudio = false,
                    speakerLabelsAvailable = true,
                    speakerLabelsEnabled = true,
                    speakerPeople = listOf(
                        SpeakerPersonUi(
                            id = "me",
                            name = "Me",
                            isSelf = true,
                            samples = listOf(SpeakerSampleUi("sample-1", "Recorded voice sample", "Thu, Nov 19 · 18 s of speech")),
                        ),
                    ),
                    onAction = actions::add,
                )
            }
        }

        composeRule.onNode(hasScrollToNodeAction()).performScrollToNode(hasTestTag("settings_speaker_person"))
        composeRule.onNodeWithText("1 voice sample").assertIsDisplayed()
        composeRule.onNode(hasScrollToNodeAction()).performScrollToNode(hasTestTag("settings_record_voice_sample"))
        composeRule.onNodeWithTag("settings_record_voice_sample").performClick()
        composeRule.onAllNodesWithTag("settings_delete_voice_sample").onFirst().performClick()

        assertEquals(
            listOf(VoiceMemoUiAction.StartVoiceSample, VoiceMemoUiAction.DeleteVoiceSample("sample-1")),
            actions,
        )
    }

    @Test
    fun settingsHideTheSpeakerSectionWhenTheBuildHasNoSpeakerModels() {
        composeRule.setContent {
            VoiceMemoTheme(dynamicColor = false) {
                VoiceMemoLicensesScreen(licenses = emptyList(), isImportingAudio = false, onAction = {})
            }
        }

        composeRule.onAllNodesWithTag("settings_speaker_labels").assertCountEquals(0)
    }
}
