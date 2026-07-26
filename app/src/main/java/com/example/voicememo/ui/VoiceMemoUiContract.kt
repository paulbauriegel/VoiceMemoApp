package com.example.voicememo.ui

/**
 * Presentation-only state consumed by [VoiceMemoApp].
 *
 * The UI never records, plays, shares, deletes, or transcribes directly. The app layer owns
 * those side effects, reduces [VoiceMemoUiAction] values, and supplies the resulting state.
 */
data class VoiceMemoAppUiState(
    val destination: VoiceMemoDestination = VoiceMemoDestination.Library,
    val query: String = "",
    val memos: List<VoiceMemoListItemUi> = emptyList(),
    val selectedMemoIds: Set<String> = emptySet(),
    val selectedMemoId: String? = null,
    val isImportingAudio: Boolean = false,
    val isExportingBackup: Boolean = false,
    val isRestoringBackup: Boolean = false,
    val showRestoreBackupConfirmation: Boolean = false,
    val transcriptionTimestampsEnabled: Boolean = false,
    val voiceActivityDetectionEnabled: Boolean = true,
    val gpuAccelerationEnabled: Boolean = true,
    val recorder: RecorderUiState = RecorderUiState(),
    val detail: VoiceMemoDetailUi? = null,
    val modelAvailability: ModelAvailabilityUiState = ModelAvailabilityUiState(),
    val pendingDelete: PendingDeleteUi? = null,
    val pendingShare: PendingShareUi? = null,
    val licenses: List<LicenseUi> = emptyList(),
)

enum class VoiceMemoDestination {
    Library,
    Recorder,
    Detail,
    Settings,
}

data class VoiceMemoListItemUi(
    val id: String,
    val title: String,
    val monthLabel: String,
    val dateLabel: String,
    val durationLabel: String,
    val waveform: List<Float>,
    val playbackFraction: Float = 0f,
    val isPlaying: Boolean = false,
    val transcriptPreview: String? = null,
    val transcription: TranscriptionSummaryUi = TranscriptionSummaryUi(),
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

    object BeginTitleEdit : VoiceMemoUiAction
    data class TitleDraftChanged(val title: String) : VoiceMemoUiAction
    object SaveTitle : VoiceMemoUiAction
    object CancelTitleEdit : VoiceMemoUiAction

    object ToggleDetailPlayback : VoiceMemoUiAction
    data class DetailSeekChanged(val fraction: Float) : VoiceMemoUiAction
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
        transcription = TranscriptionSummaryUi(
            phase = TranscriptionPhase.Transcribing,
            progress = .58f,
            statusLabel = "Transcribing 58%",
        ),
    )

    val detail = VoiceMemoDetailUi(
        id = memoOne.id,
        title = memoOne.title,
        dateLabel = memoOne.dateLabel,
        durationLabel = memoOne.durationLabel,
        playbackPositionLabel = "00:53",
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
        memos = listOf(memoOne, memoTwo),
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
