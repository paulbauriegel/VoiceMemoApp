package com.example.voicememo.ui

import com.example.voicememo.data.ProjectIcon

/**
 * Presentation-only state consumed by [VoiceMemoApp].
 *
 * The UI never records, plays, shares, deletes, or transcribes directly. The app layer owns
 * those side effects, reduces [VoiceMemoUiAction] values, and supplies the resulting state.
 */
data class VoiceMemoAppUiState(
    val destination: VoiceMemoDestination = VoiceMemoDestination.Library,
    val query: String = "",
    val libraryTab: LibraryTab = LibraryTab.Recents,
    val memos: List<VoiceMemoListItemUi> = emptyList(),
    val projects: List<ProjectListItemUi> = emptyList(),
    val activeProjectId: String? = null,
    val activeProjectName: String? = null,
    val activeProjectIcon: ProjectIcon = ProjectIcon.WORK,
    val activeProjectArchived: Boolean = false,
    val selectedMemoIds: Set<String> = emptySet(),
    val selectedMemoId: String? = null,
    val isImportingAudio: Boolean = false,
    val isExportingBackup: Boolean = false,
    val isRestoringBackup: Boolean = false,
    val showRestoreBackupConfirmation: Boolean = false,
    val transcriptionTimestampsEnabled: Boolean = false,
    val voiceActivityDetectionEnabled: Boolean = true,
    val gpuAccelerationEnabled: Boolean = true,
    val transcriptionModelId: String = "whisper-large-v3-turbo",
    val recorder: RecorderUiState = RecorderUiState(),
    val detail: VoiceMemoDetailUi? = null,
    val modelAvailability: ModelAvailabilityUiState = ModelAvailabilityUiState(),
    val pendingDelete: PendingDeleteUi? = null,
    val pendingShare: PendingShareUi? = null,
    val pendingProjectDelete: ProjectListItemUi? = null,
    val licenses: List<LicenseUi> = emptyList(),
)

enum class VoiceMemoDestination {
    Library,
    Recorder,
    Detail,
    Settings,
}

enum class LibraryTab {
    Recents,
    Projects,
}

data class VoiceMemoListItemUi(
    val id: String,
    val projectId: String? = null,
    val title: String,
    val monthLabel: String,
    val dateLabel: String,
    val durationLabel: String,
    val waveform: List<Float>,
    val playbackFraction: Float = 0f,
    val isPlaying: Boolean = false,
    val transcriptPreview: String? = null,
    val transcription: TranscriptionSummaryUi = TranscriptionSummaryUi(),
    val createdAtMillis: Long = 0L,
    val isStarred: Boolean = false,
)

data class ProjectListItemUi(
    val id: String,
    val name: String,
    val recordingCount: Int,
    val recordingCountLabel: String,
    val updatedLabel: String,
    val icon: ProjectIcon = ProjectIcon.WORK,
    val isArchived: Boolean = false,
)

data class TranscriptionSummaryUi(
    val phase: TranscriptionPhase = TranscriptionPhase.NotRequested,
    val progress: Float? = null,
    val statusLabel: String = "Not transcribed",
)

enum class RecorderPhase {
    Ready,
    Recording,
    Paused,
    Stopping,
}

data class RecorderUiState(
    val title: String = "New recording",
    val elapsedLabel: String = "00:00.0",
    val phase: RecorderPhase = RecorderPhase.Ready,
    val waveform: List<Float> = emptyList(),
    val errorMessage: String? = null,
)

enum class DetailSection {
    Audio,
    Transcript,
}

data class VoiceMemoDetailUi(
    val id: String,
    val title: String,
    val titleDraft: String = title,
    val isEditingTitle: Boolean = false,
    val dateLabel: String,
    val durationLabel: String,
    val durationMillis: Long = 0L,
    val playbackPositionLabel: String = "00:00",
    val playbackPositionMillis: Long = 0L,
    val playbackFraction: Float = 0f,
    val isPlaying: Boolean = false,
    val waveform: List<Float>,
    val selectedSection: DetailSection = DetailSection.Audio,
    val transcript: TranscriptDetailUi = TranscriptDetailUi(),
)

enum class TranscriptionPhase {
    NotRequested,
    Queued,
    Preparing,
    Transcribing,
    Finalizing,
    Complete,
    Error,
}

data class TranscriptDetailUi(
    val phase: TranscriptionPhase = TranscriptionPhase.NotRequested,
    val progress: Float? = null,
    val statusLabel: String = "Ready to transcribe locally",
    val text: String = "",
    val segments: List<TranscriptSegmentUi> = emptyList(),
    val draft: String = text,
    val isEditing: Boolean = false,
    val isEdited: Boolean = false,
    val selectedLanguageCode: String = "auto",
    val languages: List<LanguageUi> = defaultLanguages,
    val errorMessage: String? = null,
)

data class TranscriptSegmentUi(
    val id: String,
    val startMillis: Long,
    val endMillis: Long,
    val startLabel: String,
    val endLabel: String,
    val text: String,
)

data class LanguageUi(
    val code: String,
    val label: String,
)

val defaultLanguages: List<LanguageUi> = listOf(
    LanguageUi(code = "auto", label = "Auto-detect"),
    LanguageUi(code = "de", label = "German"),
    LanguageUi(code = "en", label = "English"),
    LanguageUi(code = "es", label = "Spanish"),
    LanguageUi(code = "fr", label = "French"),
)

enum class ModelAvailabilityPhase {
    Hidden,
    Checking,
    Available,
    Unsupported,
    Error,
}

data class ModelAvailabilityUiState(
    val phase: ModelAvailabilityPhase = ModelAvailabilityPhase.Hidden,
    val headline: String = "Checking local transcription",
    val supportingText: String = "Verifying the bundled Whisper model on this device.",
    val progress: Float? = null,
)

data class PendingDeleteUi(
    val memoIds: Set<String>,
) {
    val count: Int get() = memoIds.size
}

data class PendingShareUi(
    val memoIds: Set<String>,
    val transcriptAvailableForAll: Boolean,
) {
    val count: Int get() = memoIds.size
}

enum class SharePayload {
    Audio,
    Transcript,
    AudioAndTranscript,
}

data class LicenseUi(
    val name: String,
    val version: String? = null,
    val licenseName: String,
    val notice: String? = null,
    val sourceUrl: String? = null,
)

sealed interface VoiceMemoUiAction {
    data class SearchChanged(val query: String) : VoiceMemoUiAction
    data class LibraryTabChanged(val tab: LibraryTab) : VoiceMemoUiAction
    data class OpenProject(val projectId: String) : VoiceMemoUiAction
    object CloseProject : VoiceMemoUiAction
    data class CreateProject(
        val name: String,
        val memoIds: Set<String> = emptySet(),
    ) : VoiceMemoUiAction
    data class UpdateProject(
        val projectId: String,
        val name: String,
        val icon: ProjectIcon,
    ) : VoiceMemoUiAction
    data class SetProjectArchived(
        val projectId: String,
        val archived: Boolean,
    ) : VoiceMemoUiAction
    data class AssignMemosToProject(
        val memoIds: Set<String>,
        val projectId: String?,
    ) : VoiceMemoUiAction
    data class RequestDeleteProject(val projectId: String) : VoiceMemoUiAction
    object ConfirmDeleteProject : VoiceMemoUiAction
    object DismissDeleteProject : VoiceMemoUiAction

    object ImportAudio : VoiceMemoUiAction
    object ExportBackup : VoiceMemoUiAction
    object RequestRestoreBackup : VoiceMemoUiAction
    object ConfirmRestoreBackup : VoiceMemoUiAction
    object DismissRestoreBackup : VoiceMemoUiAction
    object StartRecording : VoiceMemoUiAction
    object PauseRecording : VoiceMemoUiAction
    object ResumeRecording : VoiceMemoUiAction
    object StopRecording : VoiceMemoUiAction

    data class OpenMemo(val memoId: String) : VoiceMemoUiAction
    data class ToggleMemoPlayback(val memoId: String) : VoiceMemoUiAction
    data class ToggleMemoStarred(val memoId: String) : VoiceMemoUiAction
    data class ToggleMemoSelection(val memoId: String) : VoiceMemoUiAction
    object ClearSelection : VoiceMemoUiAction
    object SelectAll : VoiceMemoUiAction

    data class RequestDelete(val memoIds: Set<String>) : VoiceMemoUiAction
    object ConfirmDelete : VoiceMemoUiAction
    object DismissDelete : VoiceMemoUiAction

    data class RequestShare(val memoIds: Set<String>) : VoiceMemoUiAction
    data class ShareAs(val payload: SharePayload) : VoiceMemoUiAction
    object DismissShare : VoiceMemoUiAction

    object NavigateBack : VoiceMemoUiAction
    object OpenSettings : VoiceMemoUiAction
    data class OpenLicenseSource(val url: String) : VoiceMemoUiAction
    object RetryModelAvailabilityCheck : VoiceMemoUiAction
    data class TranscriptionTimestampsChanged(val enabled: Boolean) : VoiceMemoUiAction
    data class VoiceActivityDetectionChanged(val enabled: Boolean) : VoiceMemoUiAction
    data class GpuAccelerationChanged(val enabled: Boolean) : VoiceMemoUiAction
    data class TranscriptionModelChanged(val modelId: String) : VoiceMemoUiAction

    object BeginTitleEdit : VoiceMemoUiAction
    data class TitleDraftChanged(val title: String) : VoiceMemoUiAction
    object SaveTitle : VoiceMemoUiAction
    object CancelTitleEdit : VoiceMemoUiAction

    object ToggleDetailPlayback : VoiceMemoUiAction
    data class DetailSeekChanged(val fraction: Float) : VoiceMemoUiAction
    data class DetailSeekBy(val deltaMillis: Long) : VoiceMemoUiAction
    data class DetailSectionChanged(val section: DetailSection) : VoiceMemoUiAction

    data class TranscriptLanguageChanged(val languageCode: String) : VoiceMemoUiAction
    object RequestTranscription : VoiceMemoUiAction
    object RetryTranscription : VoiceMemoUiAction
    object CancelTranscription : VoiceMemoUiAction
    object BeginTranscriptEdit : VoiceMemoUiAction
    data class TranscriptDraftChanged(val text: String) : VoiceMemoUiAction
    object SaveTranscript : VoiceMemoUiAction
    object CancelTranscriptEdit : VoiceMemoUiAction
    object ResetEditedTranscript : VoiceMemoUiAction
}

object VoiceMemoPreviewData {
    private val waveformOne = listOf(
        .12f, .22f, .48f, .76f, .34f, .18f, .62f, .91f, .46f, .27f, .72f, .41f,
        .15f, .58f, .83f, .33f, .67f, .24f, .43f, .79f, .52f, .20f, .64f, .36f,
    )
    private val waveformTwo = waveformOne.reversed()

    val memoOne = VoiceMemoListItemUi(
        id = "memo-1",
        title = "Project notes",
        monthLabel = "July",
        dateLabel = "Thu, Jul 23",
        durationLabel = "02:46",
        waveform = waveformOne,
        playbackFraction = .32f,
        transcriptPreview = "We should keep the first release focused on local recording and transcription.",
        createdAtMillis = System.currentTimeMillis(),
        isStarred = true,
        transcription = TranscriptionSummaryUi(
            phase = TranscriptionPhase.Complete,
            statusLabel = "Transcript ready",
        ),
    )

    val memoTwo = VoiceMemoListItemUi(
        id = "memo-2",
        title = "Jul 22 at 4:59 PM",
        monthLabel = "July",
        dateLabel = "Wed, Jul 22",
        durationLabel = "16:59",
        waveform = waveformTwo,
        createdAtMillis = System.currentTimeMillis() - 86_400_000L,
        transcription = TranscriptionSummaryUi(
            phase = TranscriptionPhase.Transcribing,
            progress = .58f,
            statusLabel = "Transcribing 58%",
        ),
    )

    val memoThree = memoTwo.copy(
        id = "memo-3",
        title = "Jul 21 at 9:18 AM",
        dateLabel = "Tue, Jul 21",
        durationLabel = "04:32",
        waveform = waveformOne,
        createdAtMillis = System.currentTimeMillis() - 172_800_000L,
    )

    val projects = listOf(
        ProjectListItemUi(
            id = "project-work",
            name = "Work",
            recordingCount = 12,
            recordingCountLabel = "12 recordings",
            updatedLabel = "Updated 2h ago",
        ),
        ProjectListItemUi(
            id = "project-ideas",
            name = "Ideas",
            recordingCount = 7,
            recordingCountLabel = "7 recordings",
            updatedLabel = "Updated yesterday",
            icon = ProjectIcon.LIGHTBULB,
        ),
        ProjectListItemUi(
            id = "project-personal",
            name = "Personal",
            recordingCount = 5,
            recordingCountLabel = "5 recordings",
            updatedLabel = "Updated Jul 20",
            icon = ProjectIcon.PERSON,
        ),
    )

    val detail = VoiceMemoDetailUi(
        id = memoOne.id,
        title = memoOne.title,
        dateLabel = memoOne.dateLabel,
        durationLabel = memoOne.durationLabel,
        durationMillis = 166_000L,
        playbackPositionLabel = "00:53",
        playbackPositionMillis = 53_000L,
        playbackFraction = .32f,
        waveform = waveformOne + waveformTwo,
        selectedSection = DetailSection.Transcript,
        transcript = TranscriptDetailUi(
            phase = TranscriptionPhase.Complete,
            statusLabel = "Transcript ready",
            text = "We should keep the first release focused on local recording and transcription. " +
                "The audio and transcript stay on this device.",
            segments = listOf(
                TranscriptSegmentUi(
                    id = "preview-0",
                    startMillis = 0L,
                    endMillis = 28_000L,
                    startLabel = "00:00",
                    endLabel = "00:28",
                    text = "We should keep the first release focused on local recording and transcription.",
                ),
                TranscriptSegmentUi(
                    id = "preview-1",
                    startMillis = 28_000L,
                    endMillis = 56_000L,
                    startLabel = "00:28",
                    endLabel = "00:56",
                    text = "The audio and transcript stay on this device.",
                ),
            ),
        ),
    )

    val appState = VoiceMemoAppUiState(
        memos = listOf(
            memoOne,
            memoTwo.copy(projectId = projects[1].id),
            memoThree.copy(projectId = projects[2].id),
        ),
        projects = projects,
        selectedMemoId = memoOne.id,
        detail = detail,
        modelAvailability = ModelAvailabilityUiState(
            phase = ModelAvailabilityPhase.Checking,
            headline = "Checking local transcription",
            supportingText = "Verifying the bundled Whisper v3 Turbo model",
        ),
        licenses = listOf(
            LicenseUi(
                name = "Whisper large-v3 Turbo",
                licenseName = "MIT",
                sourceUrl = "https://huggingface.co/litert-community/whisper-large-v3-turbo",
            ),
            LicenseUi(
                name = "AndroidX Compose Material 3",
                licenseName = "Apache License 2.0",
                sourceUrl = "https://developer.android.com/jetpack/androidx/releases/compose-material3",
            ),
        ),
    )
}
