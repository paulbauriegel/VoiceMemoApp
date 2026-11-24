package com.example.voicememo

import android.app.ActivityManager
import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.util.Log
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.voicememo.data.MemoDetail
import com.example.voicememo.data.MemoLibrarySearchResult
import com.example.voicememo.data.MemoSummary
import com.example.voicememo.data.ProjectIcon
import com.example.voicememo.data.ProjectSummary
import com.example.voicememo.data.TranscriptionJob
import com.example.voicememo.data.TranscriptionJobStatus
import com.example.voicememo.data.TranscriptChunk
import com.example.voicememo.data.SemanticHit
import com.example.voicememo.data.SemanticTranscriptRanking
import com.example.voicememo.data.TimedTranscriptSegment
import com.example.voicememo.data.TranscriptSegments
import com.example.voicememo.data.TranscriptSource
import com.example.voicememo.playback.PlaybackState
import com.example.voicememo.recording.RecordingState
import com.example.voicememo.recording.WaveformBucket
import com.example.voicememo.recording.WaveformSidecar
import com.example.voicememo.sharing.ShareContent
import com.example.voicememo.sharing.ShareableMemo
import com.example.voicememo.settings.AppearanceMode
import com.example.voicememo.embedding.MultilingualE5Embedder
import com.example.voicememo.coordination.InferenceGate
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.speaker.PyannoteSegmenter
import com.example.voicememo.speaker.SpeakerLabel
import com.example.voicememo.speaker.SpeakerProfiles
import com.example.voicememo.speaker.SpeakerTimeline
import com.example.voicememo.work.SpeakerLabelWorker
import com.example.voicememo.speaker.CamPlusPlusEmbedder
import com.example.voicememo.ui.DetailSection
import com.example.voicememo.ui.LicenseUi
import com.example.voicememo.ui.LibraryTab
import com.example.voicememo.ui.ModelAvailabilityPhase
import com.example.voicememo.ui.ModelAvailabilityUiState
import com.example.voicememo.ui.PendingDeleteUi
import com.example.voicememo.ui.PendingShareUi
import com.example.voicememo.ui.ProjectListItemUi
import com.example.voicememo.ui.RecorderPhase
import com.example.voicememo.ui.RecorderUiState
import com.example.voicememo.ui.SharePayload
import com.example.voicememo.ui.SemanticSearchUi
import com.example.voicememo.ui.TranscriptSegmentUi
import com.example.voicememo.ui.TranscriptDetailUi
import com.example.voicememo.ui.TranscriptionPhase
import com.example.voicememo.ui.TranscriptionSummaryUi
import com.example.voicememo.ui.VoiceMemoAppUiState
import com.example.voicememo.ui.VoiceMemoDestination
import com.example.voicememo.ui.VoiceMemoDetailUi
import com.example.voicememo.ui.VoiceMemoListItemUi
import com.example.voicememo.ui.VoiceMemoUiAction
import com.example.voicememo.ui.defaultLanguages
import com.example.voicememo.transcription.SileroVoiceActivityDetector
import com.example.voicememo.transcription.GraniteRecordingMetadataGenerator
import com.example.voicememo.transcription.WhisperAssets
import com.example.voicememo.transcription.ParakeetAssets
import com.example.voicememo.transcription.ParakeetModelFiles
import com.example.voicememo.transcription.WhisperTimestampTokens
import com.example.voicememo.transcription.WhisperTokenizer
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.mapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface VoiceMemoUiEffect {
    data object LaunchAudioPicker : VoiceMemoUiEffect
    data class LaunchBackupExport(val suggestedFileName: String) : VoiceMemoUiEffect
    data object LaunchBackupRestore : VoiceMemoUiEffect
    data class LaunchIntent(val intent: Intent) : VoiceMemoUiEffect
    data class Message(val text: String) : VoiceMemoUiEffect
}

class VoiceMemoViewModel(
    application: Application,
) : AndroidViewModel(application) {
    private val container = (application as VoiceMemoApplication).container
    private val repository = container.memoRepository
    private val presentation = MutableStateFlow(
        PresentationState(
            appearanceMode = container.appearancePreferences.mode,
            transcriptionTimestampsEnabled =
                container.transcriptionPreferences.timestampsEnabled,
            voiceActivityDetectionEnabled =
                container.transcriptionPreferences.voiceActivityDetectionEnabled,
            gpuAccelerationEnabled =
                container.transcriptionPreferences.gpuAccelerationEnabled,
            transcribeWhileRecordingEnabled =
                container.transcriptionPreferences.transcribeWhileRecordingEnabled,
            transcriptionTimingEnabled =
                container.transcriptionPreferences.transcriptionTimingEnabled,
            transcriptionModelId =
                container.transcriptionPreferences.transcriptionModelId,
            speakerLabelsEnabled =
                container.transcriptionPreferences.speakerLabelsEnabled,
            joinSpeakerLines =
                container.transcriptionPreferences.joinSpeakerLinesEnabled,
        ),
    )
    private val stopVoiceSample = java.util.concurrent.atomic.AtomicBoolean(false)
    private var semanticSearchJob: Job? = null
    private val waveformCache = ConcurrentHashMap<WaveformCacheKey, List<Float>>()
    private val mutableEffects = MutableSharedFlow<VoiceMemoUiEffect>(extraBufferCapacity = 8)
    val effects = mutableEffects.asSharedFlow()
    private val licenseItems: List<LicenseUi> by lazy {
        val resources = getApplication<Application>().resources
        listOf(
            LicenseUi(
                name = "OpenAI Whisper",
                version = WhisperAssets.MODEL_REVISION.take(8),
                licenseName = "MIT License",
                notice = resources.openRawResource(R.raw.openai_whisper_mit)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://huggingface.co/litert-community/whisper-large-v3-turbo",
            ),
            LicenseUi(
                name = "NVIDIA Parakeet TDT 0.6B v3",
                version = ParakeetAssets.TOKENIZER_REVISION.take(8),
                licenseName = "Creative Commons Attribution 4.0",
                notice =
                    "Model weights and tokenizer from NVIDIA. Attribution: NVIDIA " +
                        "Parakeet TDT 0.6B v3.",
                sourceUrl = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
            ),
            LicenseUi(
                name = "LiteRT Parakeet conversion",
                version = ParakeetAssets.ENCODER_SHA256.take(8),
                licenseName = "Apache License 2.0",
                notice = resources.openRawResource(R.raw.litert_apache_2_0)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3",
            ),
            LicenseUi(
                name = "Google LiteRT",
                version = "2.1.6",
                licenseName = "Apache License 2.0",
                notice = resources.openRawResource(R.raw.litert_apache_2_0)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://ai.google.dev/edge/litert",
            ),
            LicenseUi(
                name = "IBM Granite 4.0 350M",
                version = "6f1e9bce",
                licenseName = "Apache License 2.0",
                notice = resources.openRawResource(R.raw.litert_apache_2_0)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl =
                    "https://huggingface.co/litert-community/granite-4.0-350m-litert-lm",
            ),
            LicenseUi(
                name = "Google LiteRT-LM",
                version = "0.14.0",
                licenseName = "Apache License 2.0",
                notice = resources.openRawResource(R.raw.litert_apache_2_0)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://github.com/google-ai-edge/LiteRT-LM",
            ),
            LicenseUi(
                name = "Silero VAD",
                version = SileroVoiceActivityDetector.MODEL_VERSION,
                licenseName = "MIT License",
                notice = resources.openRawResource(R.raw.silero_vad_mit)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://github.com/snakers4/silero-vad",
            ),
            LicenseUi(
                name = "multilingual-e5-small",
                version = MultilingualE5Embedder.MODEL_REVISION.take(8),
                licenseName = "MIT License",
                notice = resources.openRawResource(R.raw.multilingual_e5_mit)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://huggingface.co/intfloat/multilingual-e5-small",
            ),
            LicenseUi(
                name = "pyannote segmentation",
                version = PyannoteSegmenter.MODEL_VERSION,
                licenseName = "MIT License",
                notice = resources.openRawResource(R.raw.pyannote_segmentation_mit)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://huggingface.co/pyannote/segmentation-3.0",
            ),
            LicenseUi(
                name = "3D-Speaker CAM++ (Chinese and English)",
                version = CamPlusPlusEmbedder.MODEL_SHA256.take(8),
                licenseName = "Apache License 2.0",
                notice = resources.openRawResource(R.raw.threed_speaker_campplus_apache_2_0)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://github.com/modelscope/3D-Speaker",
            ),
            LicenseUi(
                name = "Microsoft ONNX Runtime",
                version = "1.26.0",
                licenseName = "MIT License",
                notice = resources.openRawResource(R.raw.onnx_runtime_mit)
                    .bufferedReader()
                    .use { it.readText() },
                sourceUrl = "https://github.com/microsoft/onnxruntime",
            ),
        )
    }
    private val transcriptTokenizer: WhisperTokenizer by lazy {
        getApplication<Application>().assets.open(WhisperAssets.TOKENIZER).use {
            WhisperTokenizer.load(it)
        }
    }

    private val searchQueries = presentation
        .map { it.query }
        .distinctUntilChanged()
        .mapLatest { query ->
            if (query.isNotBlank()) delay(120L)
            query
        }

    private val library: Flow<MemoLibrarySearchResult> = repository
        .observeLibraryQueries(searchQueries)
        .mapLatest { result ->
            withContext(Dispatchers.IO) {
                result.memos.forEach { memo ->
                    memo.waveformPath?.let {
                        cachedWaveform(
                            path = it,
                            maximumBuckets = LIBRARY_WAVEFORM_BUCKETS,
                            generation = memo.updatedAtMillis,
                        )
                    }
                }
            }
            result
        }

    private val libraryContent: Flow<LibraryContent> = combine(
        library,
        repository.observeProjects(),
    ) { searchResult, projects ->
        LibraryContent(searchResult = searchResult, projects = projects)
    }

    private val detail: Flow<MemoDetail?> = presentation
        .map { it.selectedMemoId }
        .distinctUntilChanged()
        .flatMapLatest { memoId ->
            if (memoId == null) flowOf(null) else repository.observeMemo(memoId)
        }
        .mapLatest { memo ->
            withContext(Dispatchers.IO) {
                memo?.memo?.waveformPath?.let {
                    cachedWaveform(
                        path = it,
                        maximumBuckets = DETAIL_WAVEFORM_BUCKETS,
                        generation = memo.memo.updatedAtMillis,
                    )
                }
            }
            memo
        }

    val uiState: StateFlow<VoiceMemoAppUiState> = combine(
        libraryContent,
        detail,
        container.recorderController.state,
        container.memoPlayer.state,
        presentation,
    ) { content, memoDetail, recorder, playback, local ->
        val searchResult = content.searchResult
        val memos = searchResult.memos.takeIf {
            searchResult.query == local.query
        }.orEmpty().let { allMemos ->
            local.activeProjectId?.let { projectId ->
                allMemos.filter { it.projectId == projectId }
            } ?: allMemos
        }
        val activeProject = local.activeProjectId?.let { projectId ->
            content.projects.firstOrNull { it.id == projectId }
        }
        val visibleMemoIds = if (
            local.libraryTab == LibraryTab.Projects &&
            local.activeProjectId == null
        ) {
            emptySet()
        } else {
            memos.mapTo(hashSetOf(), MemoSummary::id)
        }
        VoiceMemoAppUiState(
            destination = local.destination,
            query = local.query,
            libraryTab = local.libraryTab,
            memos = memos.map { it.toUi(playback) },
            projects = content.projects.map { it.toUi() },
            activeProjectId = activeProject?.id,
            activeProjectName = activeProject?.name,
            activeProjectIcon = activeProject?.icon ?: ProjectIcon.WORK,
            activeProjectArchived = activeProject?.isArchived ?: false,
            selectedMemoIds = local.selectedMemoIds.intersect(visibleMemoIds),
            selectedMemoId = local.selectedMemoId,
            isImportingAudio = local.isImportingAudio || local.isAudioPickerOpen,
            isExportingBackup = local.isExportingBackup || local.isBackupExportPickerOpen,
            isRestoringBackup = local.isRestoringBackup || local.isBackupRestorePickerOpen,
            showRestoreBackupConfirmation = local.showRestoreBackupConfirmation,
            appearanceMode = local.appearanceMode,
            transcriptionTimestampsEnabled = local.transcriptionTimestampsEnabled,
            voiceActivityDetectionEnabled = local.voiceActivityDetectionEnabled,
            gpuAccelerationEnabled = local.gpuAccelerationEnabled,
            transcribeWhileRecordingEnabled = local.transcribeWhileRecordingEnabled,
            transcriptionTimingEnabled = local.transcriptionTimingEnabled,
            transcriptionModelId = local.transcriptionModelId,
            parakeetInstalled = ParakeetModelFiles.isInstalled(getApplication<Application>()),
            speakerLabelsAvailable = BuildConfig.SPEAKER_LABELS_ENABLED,
            speakerLabelsEnabled = local.speakerLabelsEnabled,
            joinSpeakerLines = local.joinSpeakerLines,
            speakerPeople = local.speakerPeople,
            voiceSampleRecording = local.voiceSampleRecording,
            recorder = recorder.toUi(),
            detail = memoDetail?.toUi(playback, local),
            modelAvailability = modelAvailability(),
            pendingDelete = local.pendingDelete,
            pendingShare = local.pendingShare,
            pendingProjectDelete = local.pendingProjectDeleteId?.let { projectId ->
                content.projects.firstOrNull { it.id == projectId }?.toUi()
            },
            licenses = licenseItems,
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000L),
        initialValue = VoiceMemoAppUiState(
            modelAvailability = modelAvailability(),
            licenses = licenseItems,
            appearanceMode = container.appearancePreferences.mode,
            transcriptionTimestampsEnabled =
                container.transcriptionPreferences.timestampsEnabled,
            voiceActivityDetectionEnabled =
                container.transcriptionPreferences.voiceActivityDetectionEnabled,
            gpuAccelerationEnabled =
                container.transcriptionPreferences.gpuAccelerationEnabled,
            transcribeWhileRecordingEnabled =
                container.transcriptionPreferences.transcribeWhileRecordingEnabled,
            transcriptionTimingEnabled =
                container.transcriptionPreferences.transcriptionTimingEnabled,
            transcriptionModelId =
                container.transcriptionPreferences.transcriptionModelId,
            parakeetInstalled = ParakeetModelFiles.isInstalled(getApplication<Application>()),
        ),
    )

    init {
        if (BuildConfig.SPEAKER_LABELS_ENABLED) observeSpeakers()
        viewModelScope.launch {
            container.recorderController.state.collect { state ->
                if (state is RecordingState.Completed) {
                    val targetProjectId = presentation.value.recordingProjectId
                    if (targetProjectId != null) {
                        runCatching {
                            repository.assignMemosToProject(setOf(state.memoId), targetProjectId)
                        }.onFailure {
                            mutableEffects.emitMessage(
                                "Recording saved, but it could not be added to the project.",
                            )
                        }
                    }
                    presentation.update {
                        it.copy(
                            destination = VoiceMemoDestination.Library,
                            selectedMemoIds = emptySet(),
                            recordingProjectId = null,
                        )
                    }
                }
            }
        }
    }

    fun onAction(action: VoiceMemoUiAction) {
        when (action) {
            VoiceMemoUiAction.ImportAudio -> {
                val current = presentation.value
                if (current.isBackupBusy()) {
                    mutableEffects.tryEmit(
                        VoiceMemoUiEffect.Message(
                            "Wait for the backup operation to finish before importing audio.",
                        ),
                    )
                } else if (!current.isImportingAudio && !current.isAudioPickerOpen) {
                    presentation.update { it.copy(isAudioPickerOpen = true) }
                    if (!mutableEffects.tryEmit(VoiceMemoUiEffect.LaunchAudioPicker)) {
                        presentation.update { it.copy(isAudioPickerOpen = false) }
                    }
                }
            }

            VoiceMemoUiAction.ExportBackup -> requestBackupExport()

            VoiceMemoUiAction.RequestRestoreBackup -> {
                if (canStartBackupOperation()) {
                    presentation.update { it.copy(showRestoreBackupConfirmation = true) }
                }
            }

            VoiceMemoUiAction.ConfirmRestoreBackup -> requestBackupRestore()

            VoiceMemoUiAction.DismissRestoreBackup -> presentation.update {
                it.copy(showRestoreBackupConfirmation = false)
            }

            is VoiceMemoUiAction.SearchChanged ->
                presentation.update { local ->
                    if (local.query == action.query) {
                        local
                    } else {
                        local.copy(
                            query = action.query,
                            selectedMemoIds = emptySet(),
                        )
                    }
                }

            is VoiceMemoUiAction.LibraryTabChanged -> presentation.update {
                it.copy(
                    libraryTab = action.tab,
                    activeProjectId = null,
                    selectedMemoIds = emptySet(),
                    query = "",
                )
            }

            is VoiceMemoUiAction.OpenProject -> {
                if (uiState.value.projects.any { it.id == action.projectId }) {
                    presentation.update {
                        it.copy(
                            libraryTab = LibraryTab.Projects,
                            activeProjectId = action.projectId,
                            selectedMemoIds = emptySet(),
                            query = "",
                        )
                    }
                }
            }

            VoiceMemoUiAction.CloseProject -> presentation.update {
                it.copy(
                    activeProjectId = null,
                    selectedMemoIds = emptySet(),
                    query = "",
                )
            }

            is VoiceMemoUiAction.CreateProject -> createProject(
                name = action.name,
                memoIds = action.memoIds,
            )

            is VoiceMemoUiAction.UpdateProject -> updateProject(
                projectId = action.projectId,
                name = action.name,
                icon = action.icon,
            )

            is VoiceMemoUiAction.SetProjectArchived -> setProjectArchived(
                projectId = action.projectId,
                archived = action.archived,
            )

            is VoiceMemoUiAction.AssignMemosToProject -> assignMemosToProject(
                memoIds = action.memoIds,
                projectId = action.projectId,
            )

            is VoiceMemoUiAction.RequestDeleteProject -> {
                if (uiState.value.projects.any { it.id == action.projectId }) {
                    presentation.update { it.copy(pendingProjectDeleteId = action.projectId) }
                }
            }

            VoiceMemoUiAction.ConfirmDeleteProject -> deleteProject()
            VoiceMemoUiAction.DismissDeleteProject -> presentation.update {
                it.copy(pendingProjectDeleteId = null)
            }

            VoiceMemoUiAction.StartRecording -> {
                if (
                    presentation.value.isImportingAudio ||
                    presentation.value.isAudioPickerOpen ||
                    presentation.value.isBackupBusy()
                ) {
                    mutableEffects.tryEmit(
                        VoiceMemoUiEffect.Message(
                            "Wait for the current file operation to finish before recording.",
                        ),
                    )
                    return
                }
                presentation.update {
                    it.copy(
                        destination = VoiceMemoDestination.Recorder,
                        selectedMemoIds = emptySet(),
                        recordingProjectId = it.activeProjectId,
                    )
                }
                val state = container.recorderController.state.value
                if (
                    state is RecordingState.Idle ||
                    state is RecordingState.Completed ||
                    state is RecordingState.Failed
                ) {
                    container.memoPlayer.stop()
                    container.recorderController.start()
                }
            }

            VoiceMemoUiAction.PauseRecording -> container.recorderController.pause()
            VoiceMemoUiAction.ResumeRecording -> container.recorderController.resume()
            VoiceMemoUiAction.StopRecording -> container.recorderController.stop()

            is VoiceMemoUiAction.OpenMemo -> presentation.update {
                it.copy(
                    destination = VoiceMemoDestination.Detail,
                    selectedMemoId = action.memoId,
                    selectedMemoIds = emptySet(),
                    titleDraft = null,
                    titleEditing = false,
                    segmentDraft = null,
                    editingSegmentId = null,
                )
            }

            is VoiceMemoUiAction.ToggleMemoPlayback -> {
                viewModelScope.launch {
                    repository.getMemo(action.memoId)?.let { memo ->
                        container.memoPlayer.playOrPause(memo.id, memo.audioPath)
                    }
                }
            }

            is VoiceMemoUiAction.ToggleMemoStarred -> {
                uiState.value.memos
                    .firstOrNull { it.id == action.memoId }
                    ?.let { memo ->
                        viewModelScope.launch {
                            runCatching {
                                repository.setMemoStarred(memo.id, !memo.isStarred)
                            }.onFailure {
                                mutableEffects.emitMessage("Could not update the starred recording.")
                            }
                        }
                    }
            }

            is VoiceMemoUiAction.ToggleMemoSelection -> {
                val visibleMemoIds = uiState.value.memos.mapTo(hashSetOf()) { it.id }
                if (action.memoId in visibleMemoIds) {
                    presentation.update { local ->
                        val selected = local.selectedMemoIds.toMutableSet().apply {
                            if (!add(action.memoId)) remove(action.memoId)
                        }
                        local.copy(selectedMemoIds = selected)
                    }
                }
            }

            VoiceMemoUiAction.ClearSelection ->
                presentation.update { it.copy(selectedMemoIds = emptySet()) }

            VoiceMemoUiAction.SelectAll -> {
                val visibleMemoIds = uiState.value.memos
                    .mapTo(linkedSetOf()) { it.id }
                presentation.update { it.copy(selectedMemoIds = visibleMemoIds) }
            }

            is VoiceMemoUiAction.RequestDelete -> {
                val visibleMemoIds = uiState.value.memos.mapTo(hashSetOf()) { it.id }
                val safeIds = action.memoIds.intersect(visibleMemoIds)
                if (safeIds.isNotEmpty()) {
                    presentation.update {
                        it.copy(pendingDelete = PendingDeleteUi(safeIds))
                    }
                }
            }

            VoiceMemoUiAction.ConfirmDelete -> confirmDelete()
            VoiceMemoUiAction.DismissDelete ->
                presentation.update { it.copy(pendingDelete = null) }

            is VoiceMemoUiAction.RequestShare -> {
                val selected = uiState.value.memos.filter { it.id in action.memoIds }
                if (selected.isNotEmpty()) {
                    presentation.update {
                        it.copy(
                            pendingShare = PendingShareUi(
                                memoIds = selected.mapTo(linkedSetOf()) { memo -> memo.id },
                                transcriptAvailableForAll = selected.all { memo ->
                                    !memo.transcriptPreview.isNullOrBlank()
                                },
                            ),
                        )
                    }
                }
            }

            is VoiceMemoUiAction.ShareAs -> share(action.payload)
            VoiceMemoUiAction.DismissShare ->
                presentation.update { it.copy(pendingShare = null) }

            VoiceMemoUiAction.NavigateBack -> navigateBack()
            VoiceMemoUiAction.OpenSettings -> presentation.update {
                it.copy(
                    destination = VoiceMemoDestination.Settings,
                    selectedMemoIds = emptySet(),
                )
            }

            is VoiceMemoUiAction.OpenLicenseSource -> mutableEffects.tryEmit(
                VoiceMemoUiEffect.LaunchIntent(
                    Intent(Intent.ACTION_VIEW, action.url.toUri()),
                ),
            )

            VoiceMemoUiAction.RetryModelAvailabilityCheck ->
                presentation.update { it.copy(modelCheckNonce = it.modelCheckNonce + 1) }

            is VoiceMemoUiAction.AppearanceModeChanged -> {
                container.appearancePreferences.mode = action.mode
                presentation.update { it.copy(appearanceMode = action.mode) }
            }

            is VoiceMemoUiAction.TranscriptionTimestampsChanged -> {
                container.transcriptionPreferences.timestampsEnabled = action.enabled
                presentation.update {
                    it.copy(transcriptionTimestampsEnabled = action.enabled)
                }
            }

            is VoiceMemoUiAction.VoiceActivityDetectionChanged -> {
                container.transcriptionPreferences.voiceActivityDetectionEnabled =
                    action.enabled
                presentation.update {
                    it.copy(voiceActivityDetectionEnabled = action.enabled)
                }
            }

            is VoiceMemoUiAction.SpeakerLabelsChanged -> {
                container.transcriptionPreferences.speakerLabelsEnabled = action.enabled
                presentation.update { it.copy(speakerLabelsEnabled = action.enabled) }
            }

            is VoiceMemoUiAction.JoinSpeakerLinesChanged -> {
                container.transcriptionPreferences.joinSpeakerLinesEnabled = action.enabled
                presentation.update { it.copy(joinSpeakerLines = action.enabled) }
            }

            VoiceMemoUiAction.StartVoiceSample -> startVoiceSample()
            VoiceMemoUiAction.StopVoiceSample -> stopVoiceSample.set(true)
            is VoiceMemoUiAction.DeleteVoiceSample -> viewModelScope.launch {
                runCatching { container.speakerProfiles.deleteSample(action.sampleId) }
                    .onFailure { mutableEffects.emitMessage("Could not delete the voice sample.") }
            }

            is VoiceMemoUiAction.SetSegmentSpeaker ->
                setSegmentSpeaker(listOf(action.segmentId) + action.joinedSegmentIds, action.speaker)
            VoiceMemoUiAction.IdentifySpeakers -> identifySpeakers()

            is VoiceMemoUiAction.GpuAccelerationChanged -> {
                container.transcriptionPreferences.gpuAccelerationEnabled =
                    action.enabled
                presentation.update {
                    it.copy(gpuAccelerationEnabled = action.enabled)
                }
            }

            is VoiceMemoUiAction.TranscribeWhileRecordingChanged -> {
                container.transcriptionPreferences.transcribeWhileRecordingEnabled =
                    action.enabled
                presentation.update {
                    it.copy(transcribeWhileRecordingEnabled = action.enabled)
                }
            }

            is VoiceMemoUiAction.TranscriptionTimingChanged -> {
                container.transcriptionPreferences.transcriptionTimingEnabled = action.enabled
                presentation.update {
                    it.copy(transcriptionTimingEnabled = action.enabled)
                }
            }

            is VoiceMemoUiAction.TranscriptionModelChanged -> {
                val modelId = com.example.voicememo.transcription.TranscriptionModel
                    .fromId(action.modelId).id
                container.transcriptionPreferences.transcriptionModelId = modelId
                presentation.update { it.copy(transcriptionModelId = modelId) }
            }

            VoiceMemoUiAction.BeginTitleEdit -> presentation.update { local ->
                local.copy(
                    titleEditing = true,
                    titleDraft = uiState.value.detail?.title.orEmpty(),
                )
            }

            is VoiceMemoUiAction.TitleDraftChanged ->
                presentation.update { it.copy(titleDraft = action.title) }

            VoiceMemoUiAction.SaveTitle -> saveTitle()
            VoiceMemoUiAction.CancelTitleEdit -> presentation.update {
                it.copy(titleEditing = false, titleDraft = null)
            }

            VoiceMemoUiAction.ToggleDetailPlayback -> {
                val memoId = presentation.value.selectedMemoId ?: return
                viewModelScope.launch {
                    repository.getMemo(memoId)?.let { memo ->
                        container.memoPlayer.playOrPause(memo.id, memo.audioPath)
                    }
                }
            }

            is VoiceMemoUiAction.DetailSeekChanged -> {
                val duration = container.memoPlayer.state.value.durationMillis
                    .takeIf { it > 0 }
                    ?: uiState.value.detail?.durationMillis
                        ?.takeIf { it > 0 }
                    ?: uiState.value.detail?.let { parseDuration(it.durationLabel) }
                    ?: 0L
                container.memoPlayer.seekTo((duration * action.fraction.coerceIn(0f, 1f)).toLong())
            }

            is VoiceMemoUiAction.DetailSeekToMillis -> {
                val memoId = presentation.value.selectedMemoId ?: return
                viewModelScope.launch {
                    if (container.memoPlayer.state.value.memoId != memoId) {
                        // Seeking only applies to the loaded memo, so start it first.
                        val memo = repository.getMemo(memoId) ?: return@launch
                        container.memoPlayer.playOrPause(memo.id, memo.audioPath)
                    }
                    container.memoPlayer.seekTo(action.positionMillis)
                }
            }

            is VoiceMemoUiAction.DetailSeekBy -> {
                val playerState = container.memoPlayer.state.value
                val detail = uiState.value.detail
                val duration = playerState.durationMillis
                    .takeIf { it > 0 }
                    ?: detail?.durationMillis
                        ?.takeIf { it > 0 }
                    ?: detail?.let { parseDuration(it.durationLabel) }
                    ?: 0L
                val position = if (playerState.memoId == detail?.id) {
                    playerState.positionMillis
                } else {
                    detail?.playbackPositionMillis ?: 0L
                }
                container.memoPlayer.seekTo(
                    (position + action.deltaMillis).coerceIn(0L, duration),
                )
            }

            is VoiceMemoUiAction.DetailSectionChanged ->
                presentation.update { it.copy(detailSection = action.section) }

            is VoiceMemoUiAction.TranscriptLanguageChanged -> setLanguage(action.languageCode)
            VoiceMemoUiAction.RequestTranscription,
            VoiceMemoUiAction.RetryTranscription,
            -> requestTranscription()

            is VoiceMemoUiAction.SelectTranscriptVersion -> {
                val memoId = presentation.value.selectedMemoId
                semanticSearchJob?.cancel()
                if (memoId != null) viewModelScope.launch {
                    runCatching { repository.selectTranscriptVersion(memoId, action.versionId) }
                        .onSuccess { presentation.update { it.copy(
                            editingSegmentId = null, segmentDraft = null, semanticSearch = null,
                        ) } }
                        .onFailure { mutableEffects.emitMessage(it.message ?: "Could not select transcript.") }
                }
            }

            is VoiceMemoUiAction.TranscribeAgain -> transcribeAgain(action.languageCode)

            VoiceMemoUiAction.CancelTranscription -> cancelTranscription()
            is VoiceMemoUiAction.BeginSegmentEdit -> presentation.update {
                it.copy(
                    editingSegmentId = action.segmentId,
                    segmentDraft = uiState.value.detail?.transcript?.segments
                        ?.firstOrNull { segment -> segment.id == action.segmentId }
                        ?.text
                        .orEmpty(),
                )
            }

            is VoiceMemoUiAction.SegmentDraftChanged ->
                presentation.update { it.copy(segmentDraft = action.text) }

            VoiceMemoUiAction.SaveSegmentEdit -> saveSegmentEdit()
            VoiceMemoUiAction.CancelSegmentEdit -> presentation.update {
                it.copy(editingSegmentId = null, segmentDraft = null)
            }

            is VoiceMemoUiAction.RevertSegment -> updateSegment(action.segmentId, text = null)
            is VoiceMemoUiAction.SearchTranscriptSemantically ->
                searchTranscriptSemantically(action.query)

            VoiceMemoUiAction.ResetEditedTranscript -> resetEditedTranscript()
        }
    }

    fun importAudio(uris: List<Uri>) {
        val requestedUris = uris.distinct()
        presentation.update { it.copy(isAudioPickerOpen = false) }
        if (requestedUris.isEmpty() || presentation.value.isImportingAudio) return
        presentation.update {
            it.copy(
                isImportingAudio = true,
                selectedMemoIds = emptySet(),
            )
        }
        viewModelScope.launch {
            val importedTitles = mutableListOf<String>()
            val failures = mutableListOf<String>()
            try {
                requestedUris.forEach { uri ->
                    try {
                        importedTitles += container.audioImporter.importAudio(uri).title
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (failure: Throwable) {
                        failures += failure.message ?: "Unsupported or unreadable audio"
                    }
                }
                if (importedTitles.isNotEmpty()) {
                    presentation.update {
                        it.copy(
                            query = "",
                            selectedMemoIds = emptySet(),
                        )
                    }
                }
                mutableEffects.emitMessage(
                    importResultMessage(
                        importedTitles = importedTitles,
                        requestedCount = requestedUris.size,
                        firstFailure = failures.firstOrNull(),
                    ),
                )
            } finally {
                presentation.update { it.copy(isImportingAudio = false) }
            }
        }
    }

    fun exportBackupTo(uri: Uri?) {
        presentation.update { it.copy(isBackupExportPickerOpen = false) }
        if (uri == null || presentation.value.isExportingBackup) return
        presentation.update { it.copy(isExportingBackup = true) }
        viewModelScope.launch {
            var completed = false
            try {
                val result = withContext(Dispatchers.IO) {
                    val output = getApplication<Application>()
                        .contentResolver
                        .openOutputStream(uri, "w")
                        ?: error("The selected backup destination could not be opened.")
                    output.buffered().use { stream ->
                        container.backupManager.exportBackup(stream)
                    }
                }
                completed = true
                mutableEffects.emitMessage(
                    if (result.memoCount == 1) {
                        "Backup exported with 1 voice memo."
                    } else {
                        "Backup exported with ${result.memoCount} voice memos."
                    },
                )
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Throwable) {
                mutableEffects.emitMessage(
                    "Couldn’t export backup. " +
                        (failure.message ?: "The destination could not be written."),
                )
            } finally {
                if (!completed) {
                    withContext(NonCancellable + Dispatchers.IO) {
                        runCatching {
                            getApplication<Application>().contentResolver.delete(uri, null, null)
                        }
                    }
                }
                presentation.update { it.copy(isExportingBackup = false) }
            }
        }
    }

    fun restoreBackupFrom(uri: Uri?) {
        presentation.update { it.copy(isBackupRestorePickerOpen = false) }
        if (uri == null || presentation.value.isRestoringBackup) return
        presentation.update { it.copy(isRestoringBackup = true) }
        container.memoPlayer.stop()
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val input = getApplication<Application>()
                        .contentResolver
                        .openInputStream(uri)
                        ?: error("The selected backup could not be opened.")
                    input.buffered().use { stream ->
                        container.backupManager.importBackup(stream)
                    }
                }
                presentation.update {
                    it.copy(
                        destination = VoiceMemoDestination.Library,
                        query = "",
                        selectedMemoId = null,
                        selectedMemoIds = emptySet(),
                    )
                }
                val restored = result.importedMemoIds.size
                val message = buildString {
                    append(
                        if (restored == 1) {
                            "Restored 1 voice memo."
                        } else {
                            "Restored $restored voice memos."
                        },
                    )
                    if (result.restoredVoiceSamples > 0) append(" Your voice samples were restored too.")
                    if (result.remappedMemoCount > 0) {
                        append(" Existing copies were kept separately.")
                    }
                }
                mutableEffects.emitMessage(message)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Throwable) {
                mutableEffects.emitMessage(
                    "Couldn’t restore backup. " +
                        (failure.message ?: "The selected file is not a valid Voice Memo backup."),
                )
            } finally {
                presentation.update { it.copy(isRestoringBackup = false) }
            }
        }
    }

    private fun requestBackupExport() {
        if (!canStartBackupOperation()) return
        presentation.update { it.copy(isBackupExportPickerOpen = true) }
        viewModelScope.launch {
            val hasMemos = try {
                repository.selectAllIds().isNotEmpty()
            } catch (cancelled: CancellationException) {
                presentation.update { it.copy(isBackupExportPickerOpen = false) }
                throw cancelled
            } catch (failure: Throwable) {
                presentation.update { it.copy(isBackupExportPickerOpen = false) }
                mutableEffects.emitMessage(
                    "Couldn’t prepare a backup. " +
                        (failure.message ?: "The memo library could not be read."),
                )
                return@launch
            }
            if (!hasMemos) {
                presentation.update { it.copy(isBackupExportPickerOpen = false) }
                mutableEffects.emitMessage("Record or import a voice memo before exporting a backup.")
                return@launch
            }
            val effect = VoiceMemoUiEffect.LaunchBackupExport(
                suggestedFileName = backupFileName(System.currentTimeMillis()),
            )
            if (!mutableEffects.tryEmit(effect)) {
                presentation.update { it.copy(isBackupExportPickerOpen = false) }
            }
        }
    }

    private fun requestBackupRestore() {
        presentation.update { it.copy(showRestoreBackupConfirmation = false) }
        if (!canStartBackupOperation()) return
        presentation.update { it.copy(isBackupRestorePickerOpen = true) }
        if (!mutableEffects.tryEmit(VoiceMemoUiEffect.LaunchBackupRestore)) {
            presentation.update { it.copy(isBackupRestorePickerOpen = false) }
        }
    }

    private fun canStartBackupOperation(): Boolean {
        val current = presentation.value
        if (
            current.isImportingAudio ||
            current.isAudioPickerOpen ||
            current.isBackupBusy()
        ) {
            mutableEffects.tryEmit(
                VoiceMemoUiEffect.Message("Wait for the current file operation to finish."),
            )
            return false
        }
        val recording = container.recorderController.state.value
        if (
            recording is RecordingState.Starting ||
            recording is RecordingState.Recording ||
            recording is RecordingState.Paused ||
            recording is RecordingState.Finalizing
        ) {
            mutableEffects.tryEmit(
                VoiceMemoUiEffect.Message("Stop the current recording before using backups."),
            )
            return false
        }
        return true
    }

    private fun confirmDelete() {
        val ids = presentation.value.pendingDelete?.memoIds.orEmpty()
        presentation.update { it.copy(pendingDelete = null) }
        if (ids.isEmpty()) return
        viewModelScope.launch {
            runCatching {
                if (container.memoPlayer.state.value.memoId in ids) container.memoPlayer.stop()
                repository.deleteMemos(ids)
            }.onSuccess {
                presentation.update { local ->
                    local.copy(
                        selectedMemoIds = emptySet(),
                        selectedMemoId = local.selectedMemoId?.takeUnless(ids::contains),
                        destination = if (local.selectedMemoId in ids) {
                            VoiceMemoDestination.Library
                        } else {
                            local.destination
                        },
                    )
                }
            }.onFailure { error ->
                mutableEffects.emitMessage(error.message ?: "The memo could not be deleted.")
            }
        }
    }

    private fun share(payload: SharePayload) {
        val ids = presentation.value.pendingShare?.memoIds.orEmpty()
        presentation.update { it.copy(pendingShare = null, selectedMemoIds = emptySet()) }
        if (ids.isEmpty()) return
        viewModelScope.launch {
            runCatching {
                val summaries = repository.observeLibrary().firstValue()
                    .filter { it.id in ids }
                val shareable = withContext(Dispatchers.IO) {
                    summaries.map { summary ->
                        val segments = if (payload != SharePayload.Audio) {
                            val detail = repository.observeMemo(summary.id).firstValue()
                            detail?.timedSegments().orEmpty() to detail?.speakerJobId()
                        } else {
                            emptyList<TranscriptSegmentUi>() to null
                        }
                        ShareableMemo(
                            id = summary.id,
                            title = summary.title,
                            audioPath = summary.audioPath,
                            effectiveTranscript = summary.effectiveTranscript.takeIf(String::isNotBlank),
                            transcriptSegments = segments.first.map(TranscriptSegmentUi::text),
                            transcriptRows = shareRows(segments.first, segments.second),
                        )
                    }
                }
                container.shareExporter.createChooser(
                    memos = shareable,
                    content = when (payload) {
                        SharePayload.Audio -> ShareContent.Audio
                        SharePayload.Transcript -> ShareContent.Transcript
                        SharePayload.AudioAndTranscript -> ShareContent.Both
                    },
                )
            }.onSuccess { intent ->
                mutableEffects.emit(VoiceMemoUiEffect.LaunchIntent(intent))
            }.onFailure { error ->
                mutableEffects.emitMessage(error.message ?: "Nothing is available to share.")
            }
        }
    }

    private fun saveTitle() {
        val memoId = presentation.value.selectedMemoId ?: return
        val title = presentation.value.titleDraft.orEmpty().trim()
        if (title.isBlank()) {
            mutableEffects.tryEmit(VoiceMemoUiEffect.Message("A memo title cannot be empty."))
            return
        }
        presentation.update { it.copy(titleEditing = false, titleDraft = null) }
        viewModelScope.launch {
            runCatching { repository.updateTitle(memoId, title) }
                .onFailure { mutableEffects.emitMessage(it.message ?: "The title could not be saved.") }
        }
    }

    private fun setLanguage(code: String) {
        val memoId = presentation.value.selectedMemoId ?: return
        viewModelScope.launch {
            runCatching {
                repository.setLanguageOverride(memoId, code.takeUnless { it == "auto" })
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "The language could not be changed.")
            }
        }
    }

    private fun transcribeAgain(languageCode: String) {
        val memoId = presentation.value.selectedMemoId ?: return
        presentation.update { it.copy(editingSegmentId = null, segmentDraft = null) }
        viewModelScope.launch {
            runCatching {
                repository.setLanguageOverride(
                    memoId,
                    languageCode.takeUnless { it == "auto" },
                )
            }.onSuccess {
                requestTranscription(languageCode)
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "Transcription could not be queued.")
            }
        }
    }

    private fun requestTranscription(languageCode: String? = null) {
        val detail = uiState.value.detail ?: return
        if (modelAvailability().phase == ModelAvailabilityPhase.Unsupported) {
            mutableEffects.tryEmit(
                VoiceMemoUiEffect.Message(modelAvailability().supportingText),
            )
            return
        }
        viewModelScope.launch {
            runCatching {
                container.transcriptionCoordinator.enqueue(
                    memoId = detail.id,
                    languageCode = languageCode ?: detail.transcript.selectedLanguageCode,
                    timestampsEnabled =
                        presentation.value.transcriptionTimestampsEnabled,
                    voiceActivityDetectionEnabled =
                        presentation.value.voiceActivityDetectionEnabled,
                    gpuAccelerationEnabled =
                        presentation.value.gpuAccelerationEnabled,
                    transcriptionModelId =
                        presentation.value.transcriptionModelId,
                )
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "Transcription could not be queued.")
            }
        }
    }

    private fun cancelTranscription() {
        val selected = presentation.value.selectedMemoId ?: return
        viewModelScope.launch {
            val job = repository.observeMemo(selected).firstValue()?.latestJob
            runCatching {
                container.transcriptionCoordinator.cancel(selected, job?.id)
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "Transcription could not be cancelled.")
            }
        }
    }

    private fun saveSegmentEdit() {
        val segmentId = presentation.value.editingSegmentId ?: return
        val draft = presentation.value.segmentDraft.orEmpty().trim()
        presentation.update { it.copy(editingSegmentId = null, segmentDraft = null) }
        updateSegment(segmentId, draft)
    }

    /** A null [text] reverts the segment to its generated wording. */
    private fun updateSegment(segmentId: String, text: String?) {
        val memoId = presentation.value.selectedMemoId ?: return
        viewModelScope.launch {
            runCatching {
                val detail = checkNotNull(repository.observeMemo(memoId).firstValue()) {
                    "The recording is no longer available."
                }
                val generated = detail.generatedSegments()
                val original = generated.firstOrNull { it.id == segmentId }
                    ?: error("This part of the transcript changed. Try again.")
                // Saving the generated wording is the same as reverting.
                val storedText = text?.takeUnless { it == original.text.trim() }
                val edits = detail.segmentEdits
                    .associate { it.segmentId to it.text }
                    .toMutableMap()
                    .apply { if (storedText == null) remove(segmentId) else put(segmentId, storedText) }
                val effectiveText = generated
                    .map { segment -> edits[segment.id] ?: segment.text }
                    .map(String::trim)
                    .filter(String::isNotEmpty)
                    .joinToString(separator = " ")
                repository.saveSegmentEdit(memoId, segmentId, storedText, effectiveText)
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "The transcript could not be saved.")
            }
        }
    }

    private fun searchTranscriptSemantically(query: String) {
        semanticSearchJob?.cancel()
        val memoId = presentation.value.selectedMemoId
        if (
            !BuildConfig.SEMANTIC_SEARCH_ENABLED ||
            memoId == null ||
            query.trim().length < SemanticTranscriptRanking.MIN_QUERY_LENGTH
        ) {
            presentation.update { it.copy(semanticSearch = null) }
            return
        }
        semanticSearchJob = viewModelScope.launch {
            try {
                val detail = repository.observeMemo(memoId).firstValue() ?: return@launch
                val segments = withContext(Dispatchers.Default) {
                    TranscriptSegments.timed(detail, transcriptTokenizer)
                }
                val hits = container.semanticTranscriptIndex.search(memoId, segments, query)
                presentation.update {
                    it.copy(
                        semanticSearch = SemanticSearchState(
                            memoId = memoId,
                            result = SemanticSearchUi(query, hits.map(SemanticHit::segmentId)),
                        ),
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Throwable) {
                // Fuzzy matches still work; semantic search is an addition.
                Log.w("VoiceMemoViewModel", "Semantic transcript search failed.", failure)
            }
        }
    }

    private fun resetEditedTranscript() {
        val memoId = presentation.value.selectedMemoId ?: return
        viewModelScope.launch {
            runCatching { repository.resetEditedTranscript(memoId) }
                .onFailure { mutableEffects.emitMessage(it.message ?: "The generated transcript could not be restored.") }
        }
    }

    private fun navigateBack() {
        val local = presentation.value
        when {
            local.selectedMemoIds.isNotEmpty() ->
                presentation.update { it.copy(selectedMemoIds = emptySet()) }

            local.destination == VoiceMemoDestination.Library && local.activeProjectId != null ->
                presentation.update {
                    it.copy(activeProjectId = null, query = "")
                }

            local.destination != VoiceMemoDestination.Library ->
                presentation.update {
                    it.copy(
                        destination = VoiceMemoDestination.Library,
                        titleEditing = false,
                        titleDraft = null,
                        editingSegmentId = null,
                        segmentDraft = null,
                    )
                }
        }
    }

    private fun MemoSummary.toUi(playback: PlaybackState): VoiceMemoListItemUi {
        val isCurrent = playback.memoId == id
        val displayTranscript = if (transcriptSource == TranscriptSource.GENERATED) {
            WhisperTimestampTokens.withoutTimestampMarkers(effectiveTranscript)
        } else {
            effectiveTranscript
        }
        val waveform = waveformPath
            ?.let {
                cachedWaveform(
                    path = it,
                    maximumBuckets = LIBRARY_WAVEFORM_BUCKETS,
                    generation = updatedAtMillis,
                )
            }
            .orEmpty()
        return VoiceMemoListItemUi(
            id = id,
            projectId = projectId,
            title = title,
            monthLabel = monthFormat.format(Date(createdAtMillis)),
            dateLabel = dateFormat.format(Date(createdAtMillis)),
            durationLabel = formatDuration(durationMillis),
            waveform = waveform,
            playbackFraction = if (isCurrent && playback.durationMillis > 0) {
                playback.positionMillis.toFloat() / playback.durationMillis
            } else {
                0f
            },
            isPlaying = isCurrent && playback.isPlaying,
            transcriptPreview = displayTranscript.takeIf(String::isNotBlank),
            generatedSummary = generatedSummary,
            transcription = transcriptionSummary(latestJob, displayTranscript),
            createdAtMillis = createdAtMillis,
            isStarred = isStarred,
        )
    }

    private fun ProjectSummary.toUi(): ProjectListItemUi = ProjectListItemUi(
        id = id,
        name = name,
        recordingCount = recordingCount,
        recordingCountLabel = when (recordingCount) {
            1 -> "1 recording"
            else -> "$recordingCount recordings"
        },
        updatedLabel = "Updated ${formatProjectUpdate(updatedAtMillis)}",
        icon = icon,
        isArchived = isArchived,
    )

    private fun createProject(name: String, memoIds: Set<String>) {
        viewModelScope.launch {
            runCatching {
                val project = repository.createProject(name)
                if (memoIds.isNotEmpty()) {
                    repository.assignMemosToProject(memoIds, project.id)
                }
                project
            }.onSuccess { project ->
                presentation.update {
                    it.copy(
                        libraryTab = LibraryTab.Projects,
                        activeProjectId = project.id,
                        selectedMemoIds = emptySet(),
                        query = "",
                    )
                }
            }.onFailure {
                mutableEffects.emitMessage(it.message ?: "The project could not be created.")
            }
        }
    }

    private fun updateProject(projectId: String, name: String, icon: ProjectIcon) {
        viewModelScope.launch {
            runCatching { repository.updateProject(projectId, name, icon) }
                .onFailure {
                    mutableEffects.emitMessage(it.message ?: "The project could not be updated.")
                }
        }
    }

    private fun setProjectArchived(projectId: String, archived: Boolean) {
        viewModelScope.launch {
            runCatching { repository.setProjectArchived(projectId, archived) }
                .onSuccess {
                    if (archived) {
                        presentation.update { local ->
                            local.copy(
                                activeProjectId = local.activeProjectId
                                    .takeUnless { it == projectId },
                                selectedMemoIds = emptySet(),
                                query = "",
                            )
                        }
                    }
                }
                .onFailure {
                    mutableEffects.emitMessage(
                        it.message ?: if (archived) {
                            "The project could not be archived."
                        } else {
                            "The project could not be restored."
                        },
                    )
                }
        }
    }

    private fun assignMemosToProject(memoIds: Set<String>, projectId: String?) {
        val visibleMemoIds = uiState.value.memos.mapTo(hashSetOf()) { it.id }
        val safeIds = memoIds.intersect(visibleMemoIds)
        if (safeIds.isEmpty()) return
        viewModelScope.launch {
            runCatching { repository.assignMemosToProject(safeIds, projectId) }
                .onSuccess {
                    presentation.update { it.copy(selectedMemoIds = emptySet()) }
                }
                .onFailure {
                    mutableEffects.emitMessage(
                        it.message ?: "The recordings could not be moved.",
                    )
                }
        }
    }

    private fun deleteProject() {
        val projectId = presentation.value.pendingProjectDeleteId ?: return
        presentation.update { it.copy(pendingProjectDeleteId = null) }
        viewModelScope.launch {
            runCatching { repository.deleteProject(projectId) }
                .onSuccess {
                    presentation.update {
                        it.copy(
                            activeProjectId = it.activeProjectId.takeUnless { id -> id == projectId },
                            selectedMemoIds = emptySet(),
                        )
                    }
                }
                .onFailure {
                    mutableEffects.emitMessage(it.message ?: "The project could not be deleted.")
                }
        }
    }

    private fun formatProjectUpdate(updatedAtMillis: Long): String {
        val elapsed = (System.currentTimeMillis() - updatedAtMillis).coerceAtLeast(0L)
        return when {
            elapsed < 60_000L -> "just now"
            elapsed < 3_600_000L -> "${elapsed / 60_000L}m ago"
            elapsed < 86_400_000L -> "${elapsed / 3_600_000L}h ago"
            elapsed < 172_800_000L -> "yesterday"
            else -> projectDateFormat.format(Date(updatedAtMillis))
        }
    }

    private fun MemoDetail.toUi(
        playback: PlaybackState,
        local: PresentationState,
    ): VoiceMemoDetailUi {
        val isCurrent = playback.memoId == memo.id
        val displayTranscript = if (transcriptSource == TranscriptSource.GENERATED) {
            WhisperTimestampTokens.withoutTimestampMarkers(effectiveTranscript)
        } else {
            effectiveTranscript
        }
        val displayJob = latestJob?.takeUnless { it.status == TranscriptionJobStatus.SUCCEEDED }
            ?: selectedJob ?: latestJob
        val transcriptState = transcriptionDetail(displayJob, displayTranscript)
        val timingLabel = transcriptionTimingLabel(
            enabled = local.transcriptionTimingEnabled,
            phase = transcriptState.phase,
            transcriptionElapsedMillis = displayJob?.transcriptionElapsedMillis,
            segmentInferenceMillis = chunks.mapNotNull(TranscriptChunk::inferenceMillis),
        )
        val thermalStatusLabel = transcriptionThermalStatusLabel(
            enabled = local.transcriptionTimingEnabled,
            phase = transcriptState.phase,
            maximumThermalStatus = displayJob?.maximumThermalStatus,
        )
        val realTimeFactorLabel = transcriptionRealTimeFactorLabel(
            enabled = local.transcriptionTimingEnabled,
            phase = transcriptState.phase,
            transcriptionElapsedMillis = displayJob?.transcriptionElapsedMillis,
            segmentInferenceMillis = chunks.mapNotNull(TranscriptChunk::inferenceMillis),
            audioDurationMillis = memo.durationMillis,
        )
        return VoiceMemoDetailUi(
            id = memo.id,
            title = memo.title,
            titleDraft = local.titleDraft ?: memo.title,
            isEditingTitle = local.titleEditing,
            dateLabel = dateFormat.format(Date(memo.createdAtMillis)),
            durationLabel = formatDuration(memo.durationMillis),
            durationMillis = memo.durationMillis,
            playbackPositionLabel = formatDuration(
                if (isCurrent) playback.positionMillis else 0L,
            ),
            playbackPositionMillis = if (isCurrent) playback.positionMillis else 0L,
            playbackFraction = if (isCurrent && playback.durationMillis > 0) {
                playback.positionMillis.toFloat() / playback.durationMillis
            } else {
                0f
            },
            isPlaying = isCurrent && playback.isPlaying,
            waveform = memo.waveformPath
                ?.let {
                    cachedWaveform(
                        path = it,
                        maximumBuckets = DETAIL_WAVEFORM_BUCKETS,
                        generation = memo.updatedAtMillis,
                    )
                }
                .orEmpty(),
            generatedSummary = memo.generatedSummary,
            isGeneratingMetadata = latestJob.isGeneratingRecordingMetadata(),
            selectedSection = local.detailSection,
            transcript = transcriptState.copy(
                versions = versions.mapIndexed { index, version ->
                    com.example.voicememo.ui.TranscriptVersionUi(
                        version.id,
                        "Version ${versions.size - index} · ${version.modelId?.let { id ->
                            com.example.voicememo.transcription.TranscriptionModel.entries
                                .firstOrNull { it.id == id }?.displayName ?: id
                        } ?: "Saved transcript"}" +
                            (version.transcript.generatedAtMillis?.let { " · ${dateFormat.format(Date(it))}" } ?: ""),
                        version.selected,
                    )
                },
                editingSegmentId = local.editingSegmentId,
                semanticSearch = local.semanticSearch
                    ?.takeIf { it.memoId == memo.id }
                    ?.result,
                segmentDraft = local.segmentDraft.orEmpty(),
                selectedLanguageCode = memo.languageOverride ?: "auto",
                languageLabel = transcriptLanguageLabel(
                    override = memo.languageOverride,
                    detected = transcript?.detectedLanguage
                        ?: chunks.firstNotNullOfOrNull(TranscriptChunk::detectedLanguage),
                ),
                isEdited = transcript?.editedText != null,
                isPlainTextEdit = isPlainTextEdit(),
                timingLabel = timingLabel,
                thermalStatusLabel = thermalStatusLabel,
                realTimeFactorLabel = realTimeFactorLabel,
                segments = timedSegments().withSpeakers(local, speakerJobId()),
                speakerStatus = speakerStatus(local, transcriptState.phase),
                speakerProgress = local.speakers?.takeIf { it.jobId == speakerJobId() }?.progress,
                joinSpeakerLines = local.joinSpeakerLines,
            ),
        )
    }

    private fun transcriptionSummary(
        job: TranscriptionJob?,
        transcript: String,
    ): TranscriptionSummaryUi {
        val detail = transcriptionDetail(job, transcript)
        return TranscriptionSummaryUi(detail.phase, detail.progress, detail.statusLabel)
    }

    private fun transcriptionDetail(
        job: TranscriptionJob?,
        transcript: String,
    ): TranscriptDetailUi {
        if (job == null) {
            return TranscriptDetailUi(
                phase = if (transcript.isBlank()) TranscriptionPhase.NotRequested else TranscriptionPhase.Complete,
                statusLabel = if (transcript.isBlank()) "Ready to transcribe locally" else "Transcript ready",
                text = transcript,
            )
        }
        val progress = if (job.totalChunks > 0) {
            (
                job.completedChunks +
                    job.currentToken.toFloat() / job.currentTokenLimit.coerceAtLeast(1)
                ) / job.totalChunks
        } else {
            null
        }
        val phase = when (job.status) {
            TranscriptionJobStatus.QUEUED -> TranscriptionPhase.Queued
            TranscriptionJobStatus.RUNNING,
            TranscriptionJobStatus.PAUSED_FOR_RECORDING,
            -> when (job.phase) {
                com.example.voicememo.data.TranscriptionPhase.MERGING,
                com.example.voicememo.data.TranscriptionPhase.SAVING,
                com.example.voicememo.data.TranscriptionPhase.GENERATING_METADATA,
                -> TranscriptionPhase.Finalizing
                com.example.voicememo.data.TranscriptionPhase.ENCODING,
                com.example.voicememo.data.TranscriptionPhase.DECODING,
                -> TranscriptionPhase.Transcribing
                else -> TranscriptionPhase.Preparing
            }
            TranscriptionJobStatus.SUCCEEDED -> TranscriptionPhase.Complete
            TranscriptionJobStatus.FAILED -> TranscriptionPhase.Error
            TranscriptionJobStatus.CANCEL_REQUESTED,
            TranscriptionJobStatus.CANCELLED,
            -> if (transcript.isBlank()) TranscriptionPhase.NotRequested else TranscriptionPhase.Complete
        }
        val status = when {
            job.status == TranscriptionJobStatus.PAUSED_FOR_RECORDING -> "Paused while recording"
            phase == TranscriptionPhase.Queued -> "Queued for local transcription"
            job.phase == com.example.voicememo.data.TranscriptionPhase.LOADING_MODEL ->
                "Loading local model"
            job.phase == com.example.voicememo.data.TranscriptionPhase.PREPARING_AUDIO ->
                segmentStatus("Preparing", job)
            job.phase == com.example.voicememo.data.TranscriptionPhase.ENCODING ->
                segmentStatus("Analyzing", job)
            phase == TranscriptionPhase.Preparing -> "Preparing local transcription"
            phase == TranscriptionPhase.Transcribing -> segmentStatus("Transcribing", job)
            phase == TranscriptionPhase.Finalizing -> "Finalizing transcript"
            phase == TranscriptionPhase.Complete -> job.warning ?: "Transcript ready"
            phase == TranscriptionPhase.Error -> job.errorMessage ?: "Transcription failed"
            else -> "Ready to transcribe locally"
        }
        return TranscriptDetailUi(
            phase = phase,
            progress = progress?.coerceIn(0f, 1f),
            statusLabel = status,
            text = transcript,
            errorMessage = job.errorMessage,
        )
    }

    private fun segmentStatus(action: String, job: TranscriptionJob): String {
        if (job.totalChunks <= 0) return action
        val segment = (job.currentChunkIndex ?: job.completedChunks)
            .coerceIn(0, job.totalChunks - 1)
        return "$action segment ${segment + 1} of ${job.totalChunks}"
    }

    private fun RecordingState.toUi(): RecorderUiState = when (this) {
        RecordingState.Idle -> RecorderUiState()
        RecordingState.Starting -> RecorderUiState(
            phase = RecorderPhase.Recording,
            elapsedLabel = "00:00.0",
        )
        is RecordingState.Recording -> RecorderUiState(
            phase = RecorderPhase.Recording,
            elapsedLabel = formatDurationTenths(elapsedMillis),
            waveform = waveform.normalized(),
        )
        is RecordingState.Paused -> RecorderUiState(
            phase = RecorderPhase.Paused,
            elapsedLabel = formatDurationTenths(elapsedMillis),
            waveform = waveform.normalized(),
            errorMessage = if (interrupted) "Paused because another app needed audio." else null,
        )
        is RecordingState.Finalizing -> RecorderUiState(
            phase = RecorderPhase.Stopping,
            elapsedLabel = formatDurationTenths(elapsedMillis),
        )
        is RecordingState.Completed -> RecorderUiState()
        is RecordingState.Failed -> RecorderUiState(errorMessage = message)
    }

    private fun List<WaveformBucket>.normalized(): List<Float> =
        takeLast(180).map { bucket ->
            maxOf(
                kotlin.math.abs(bucket.minimum.toInt()),
                kotlin.math.abs(bucket.maximum.toInt()),
            ) / Short.MAX_VALUE.toFloat()
        }

    private fun cachedWaveform(
        path: String,
        maximumBuckets: Int,
        generation: Long,
    ): List<Float> =
        waveformCache.getOrPut(WaveformCacheKey(path, maximumBuckets, generation)) {
            WaveformSidecar.normalized(File(path), maximumBuckets)
        }

    private fun modelAvailability(): ModelAvailabilityUiState {
        val app = getApplication<Application>()
        if (app.getSystemService(ActivityManager::class.java).isLowRamDevice) {
            return ModelAvailabilityUiState(
                phase = ModelAvailabilityPhase.Unsupported,
                headline = "Transcription isn’t supported on this device",
                supportingText =
                    "Android designates this as a low-RAM device. Recording and playback still work, but the bundled model cannot be loaded safely.",
            )
        }
        return runCatching {
            if (WhisperAssets.SPLIT_MODEL_PRODUCTION) {
                app.assets.openFd(WhisperAssets.ENCODER_MODEL).use { descriptor ->
                    check(descriptor.declaredLength == WhisperAssets.ENCODER_MODEL_SIZE_BYTES)
                }
                app.assets.openFd(WhisperAssets.DECODER_MODEL).use { descriptor ->
                    check(descriptor.declaredLength == WhisperAssets.DECODER_MODEL_SIZE_BYTES)
                }
            } else {
                app.assets.openFd(WhisperAssets.MODEL).use { descriptor ->
                    check(descriptor.declaredLength == WhisperAssets.MODEL_SIZE_BYTES)
                }
            }
            app.assets.openFd(GraniteRecordingMetadataGenerator.MODEL_ASSET).use { descriptor ->
                check(
                    descriptor.declaredLength ==
                        GraniteRecordingMetadataGenerator.MODEL_SIZE_BYTES,
                )
            }
            ModelAvailabilityUiState(
                phase = ModelAvailabilityPhase.Available,
                headline = "Local transcription ready",
                supportingText =
                    "Whisper and Granite are bundled, Parakeet is installed separately, " +
                        "and all of them run entirely on this device.",
            )
        }.getOrElse { error ->
            ModelAvailabilityUiState(
                phase = ModelAvailabilityPhase.Error,
                headline = "Local model unavailable",
                supportingText = error.message ?: "The bundled model could not be verified.",
            )
        }
    }

    private fun importResultMessage(
        importedTitles: List<String>,
        requestedCount: Int,
        firstFailure: String?,
    ): String = when {
        importedTitles.size == 1 && requestedCount == 1 ->
            "Imported “${importedTitles.single()}”."

        importedTitles.size == requestedCount ->
            "Imported ${importedTitles.size} voice memos."

        importedTitles.isNotEmpty() ->
            "Imported ${importedTitles.size} of $requestedCount voice memos. " +
                (firstFailure ?: "Some files could not be imported.")

        else -> "Couldn’t import audio. " +
            (firstFailure ?: "The selected files were unsupported or unreadable.")
    }

    private suspend fun <T> Flow<T>.firstValue(): T = first()

    private suspend fun MutableSharedFlow<VoiceMemoUiEffect>.emitMessage(message: String) {
        emit(VoiceMemoUiEffect.Message(message))
    }

    private data class PresentationState(
        val destination: VoiceMemoDestination = VoiceMemoDestination.Library,
        val query: String = "",
        val libraryTab: LibraryTab = LibraryTab.Recents,
        val activeProjectId: String? = null,
        val recordingProjectId: String? = null,
        val selectedMemoIds: Set<String> = emptySet(),
        val selectedMemoId: String? = null,
        val isImportingAudio: Boolean = false,
        val isAudioPickerOpen: Boolean = false,
        val isExportingBackup: Boolean = false,
        val isRestoringBackup: Boolean = false,
        val isBackupExportPickerOpen: Boolean = false,
        val isBackupRestorePickerOpen: Boolean = false,
        val showRestoreBackupConfirmation: Boolean = false,
        val detailSection: DetailSection = DetailSection.Audio,
        val titleDraft: String? = null,
        val titleEditing: Boolean = false,
        val editingSegmentId: String? = null,
        val segmentDraft: String? = null,
        val semanticSearch: SemanticSearchState? = null,
        val pendingDelete: PendingDeleteUi? = null,
        val pendingShare: PendingShareUi? = null,
        val pendingProjectDeleteId: String? = null,
        val modelCheckNonce: Int = 0,
        val appearanceMode: AppearanceMode = AppearanceMode.SYSTEM,
        val transcriptionTimestampsEnabled: Boolean = false,
        val voiceActivityDetectionEnabled: Boolean = true,
        val gpuAccelerationEnabled: Boolean = true,
        val transcribeWhileRecordingEnabled: Boolean = false,
        val transcriptionTimingEnabled: Boolean = false,
        val transcriptionModelId: String = "whisper-large-v3-turbo",
        val speakerLabelsEnabled: Boolean = false,
        val joinSpeakerLines: Boolean = false,
        val speakerPeople: List<com.example.voicememo.ui.SpeakerPersonUi> = emptyList(),
        val voiceSampleRecording: com.example.voicememo.ui.VoiceSampleRecordingUi =
            com.example.voicememo.ui.VoiceSampleRecordingUi(),
        val speakers: SpeakerState? = null,
    ) {
        fun isBackupBusy(): Boolean =
            isExportingBackup ||
                isRestoringBackup ||
                isBackupExportPickerOpen ||
                isBackupRestorePickerOpen
    }

    /** Speaker analysis of the transcription run whose segments are shown. */
    private class SpeakerState(
        val jobId: String,
        val analysed: Boolean,
        val timeline: SpeakerTimeline?,
        val overrides: Map<String, SpeakerLabel>,
        /** Queued or running in the background. */
        val analysing: Boolean,
        /** 0..1 once the analysis is actually running. */
        val progress: Float?,
    )

    private data class SemanticSearchState(
        val memoId: String,
        val result: SemanticSearchUi,
    )

    private data class WaveformCacheKey(
        val path: String,
        val maximumBuckets: Int,
        val generation: Long,
    )

    private val monthFormat = SimpleDateFormat("MMMM", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("EEE, MMM d", Locale.getDefault())
    private val projectDateFormat = SimpleDateFormat("MMM d", Locale.getDefault())

    private data class LibraryContent(
        val searchResult: MemoLibrarySearchResult,
        val projects: List<ProjectSummary>,
    )

    private companion object {
        const val LIBRARY_WAVEFORM_BUCKETS = 160
        const val DETAIL_WAVEFORM_BUCKETS = 260
        const val VOICE_SAMPLE_GATE = "voice-sample"
    }

    /** The transcription run whose segments are shown; speaker data is stored per run. */
    private fun MemoDetail.speakerJobId(): String? =
        selectedJob?.id ?: latestJob?.takeIf { it.status == TranscriptionJobStatus.SUCCEEDED }?.id

    private fun observeSpeakers() {
        val dao = container.database.voiceMemoDao()
        // Voice samples recorded with an earlier voice model are brought up to date once.
        viewModelScope.launch { runCatching { container.speakerProfiles.upgradeRecordedSamples() } }
        viewModelScope.launch {
            combine(dao.observeSpeakerPersons(), dao.observeSpeakerVoiceSamples()) { persons, samples ->
                persons.map { person ->
                    com.example.voicememo.ui.SpeakerPersonUi(
                        id = person.id,
                        name = person.name,
                        isSelf = person.isSelf,
                        samples = samples
                            .filter { it.personId == person.id && it.modelId == CamPlusPlusEmbedder.MODEL_ID }
                            .map { sample ->
                                val recorded = sample.origin == SpeakerVoiceSample.ORIGIN_RECORDED
                                com.example.voicememo.ui.SpeakerSampleUi(
                                    id = sample.id,
                                    title = if (recorded) "Recorded voice sample" else "From a transcript line",
                                    detail = "${dateFormat.format(Date(sample.createdAtMillis))} · " +
                                        "${(sample.speechMillis + 500) / 1_000} s of speech",
                                )
                            },
                    )
                }
            }.collect { people -> presentation.update { it.copy(speakerPeople = people) } }
        }
        viewModelScope.launch {
            detail
                .map { it?.speakerJobId() }
                .distinctUntilChanged()
                .flatMapLatest { jobId ->
                    if (jobId == null) {
                        flowOf(null)
                    } else {
                        combine(
                            dao.observeSpeakerAnalysis(jobId),
                            dao.observeSpeakerSegmentOverrides(jobId),
                            container.workManager.getWorkInfosByTagFlow(SpeakerLabelWorker.jobTag(jobId)),
                            container.speakerLabelIndex.progress,
                        ) { analysis, overrides, work, running ->
                            val progress = running?.takeIf { it.jobId == jobId }?.fraction
                            SpeakerState(
                                jobId = jobId,
                                // An analysis by an earlier voice model keeps its labels but can be redone.
                                analysed = analysis?.modelId == CamPlusPlusEmbedder.MODEL_ID,
                                timeline = analysis?.timeline?.let(SpeakerTimeline::decode),
                                overrides = overrides.associate { override ->
                                    override.segmentId to SpeakerLabel.valueOf(override.label)
                                },
                                // Also true while a new transcription labels itself.
                                analysing = progress != null || work.any { !it.state.isFinished },
                                progress = progress,
                            )
                        }
                    }
                }
                .collect { speakers -> presentation.update { it.copy(speakers = speakers) } }
        }
    }

    private fun List<TranscriptSegmentUi>.withSpeakers(
        local: PresentationState,
        jobId: String?,
    ): List<TranscriptSegmentUi> {
        val speakers = local.speakers?.takeIf { local.speakerLabelsEnabled && it.jobId == jobId } ?: return this
        return map { segment ->
            val chosen = speakers.overrides[segment.id]
            val label = chosen ?: speakers.timeline?.labelFor(segment.startMillis, segment.endMillis)
            segment.copy(
                speaker = when (label) {
                    SpeakerLabel.ME -> com.example.voicememo.ui.SpeakerUi.Me
                    SpeakerLabel.OTHER -> com.example.voicememo.ui.SpeakerUi.Other
                    SpeakerLabel.MIXED -> com.example.voicememo.ui.SpeakerUi.Mixed
                    SpeakerLabel.UNKNOWN, null -> null
                },
                speakerSetByUser = chosen != null,
            )
        }
    }

    private fun MemoDetail.speakerStatus(
        local: PresentationState,
        phase: TranscriptionPhase,
    ): com.example.voicememo.ui.SpeakerLabelStatus {
        val jobId = speakerJobId()
        if (!BuildConfig.SPEAKER_LABELS_ENABLED || !local.speakerLabelsEnabled ||
            jobId == null || phase != TranscriptionPhase.Complete || isPlainTextEdit()
        ) {
            return com.example.voicememo.ui.SpeakerLabelStatus.Hidden
        }
        val speakers = local.speakers?.takeIf { it.jobId == jobId }
        return when {
            speakers == null -> com.example.voicememo.ui.SpeakerLabelStatus.Hidden
            speakers.analysing -> com.example.voicememo.ui.SpeakerLabelStatus.Analysing
            !speakers.analysed -> com.example.voicememo.ui.SpeakerLabelStatus.NotAnalysed
            speakers.timeline == null -> com.example.voicememo.ui.SpeakerLabelStatus.NoVoiceSample
            else -> com.example.voicememo.ui.SpeakerLabelStatus.Labelled
        }
    }

    /** Timed lines with their speaker for the CSV export; speakers stay empty when labels are off. */
    private suspend fun shareRows(
        segments: List<TranscriptSegmentUi>,
        jobId: String?,
    ): List<com.example.voicememo.sharing.ShareTranscriptRow> {
        val labelled = BuildConfig.SPEAKER_LABELS_ENABLED && jobId != null &&
            container.transcriptionPreferences.speakerLabelsEnabled
        val dao = container.database.voiceMemoDao()
        val timeline = if (labelled) dao.getSpeakerAnalysis(jobId!!)?.timeline?.let(SpeakerTimeline::decode) else null
        val overrides = if (labelled) {
            dao.getSpeakerSegmentOverrides(jobId!!).associate { it.segmentId to SpeakerLabel.valueOf(it.label) }
        } else {
            emptyMap()
        }
        return segments.map { segment ->
            val label = overrides[segment.id] ?: timeline?.labelFor(segment.startMillis, segment.endMillis)
            com.example.voicememo.sharing.ShareTranscriptRow(
                startMillis = segment.startMillis,
                endMillis = segment.endMillis,
                speaker = when (label) {
                    SpeakerLabel.ME -> "Me"
                    SpeakerLabel.OTHER -> "Other"
                    SpeakerLabel.MIXED -> "Both"
                    SpeakerLabel.UNKNOWN, null -> null
                },
                text = segment.text,
            )
        }
    }

    private fun identifySpeakers() {
        val memoId = presentation.value.selectedMemoId ?: return
        val jobId = presentation.value.speakers?.jobId ?: return
        SpeakerLabelWorker.enqueue(container.workManager, memoId, jobId)
    }

    /** [segmentIds] are one line, or the lines shown joined as one. */
    private fun setSegmentSpeaker(segmentIds: List<String>, speaker: com.example.voicememo.ui.SpeakerUi?) {
        val memoId = presentation.value.selectedMemoId ?: return
        val speakers = presentation.value.speakers ?: return
        val segments = uiState.value.detail?.transcript?.segments
            ?.filter { it.id in segmentIds }
            ?.takeIf { it.isNotEmpty() }
            ?: return
        viewModelScope.launch {
            runCatching {
                val profiles = container.speakerProfiles
                when (speaker) {
                    com.example.voicememo.ui.SpeakerUi.Me -> {
                        var learned = false
                        for (segment in segments) {
                            val result = profiles.addSegmentSample(
                                personId = profiles.self().id,
                                memoId = memoId,
                                jobId = speakers.jobId,
                                segmentId = segment.id,
                                startMillis = segment.startMillis,
                                endMillis = segment.endMillis,
                            )
                            if (result is SpeakerProfiles.SampleResult.Added) learned = true
                        }
                        mutableEffects.emitMessage(
                            when {
                                learned -> "Marked as you and added to your voice samples."
                                segments.size > 1 ->
                                    "Marked as you. These lines are too short to learn your voice from."
                                else -> "Marked as you. This line is too short to learn your voice from."
                            },
                        )
                    }

                    com.example.voicememo.ui.SpeakerUi.Other, com.example.voicememo.ui.SpeakerUi.Mixed ->
                        segments.forEach { segment ->
                            profiles.setSegmentLabel(memoId, speakers.jobId, segment.id, SpeakerLabel.OTHER)
                        }

                    null -> segments.forEach { segment ->
                        profiles.setSegmentLabel(memoId, speakers.jobId, segment.id, null)
                    }
                }
            }.onFailure { mutableEffects.emitMessage("Could not change the speaker of this line.") }
        }
    }

    private fun startVoiceSample() {
        val recorder = container.voiceSampleRecorder
        when {
            presentation.value.voiceSampleRecording.let { it.isRecording || it.isProcessing } -> return
            InferenceGate.isRecording -> {
                viewModelScope.launch { mutableEffects.emitMessage("Finish the current recording first.") }
                return
            }

            !recorder.hasPermission() -> {
                viewModelScope.launch {
                    mutableEffects.emitMessage(
                        "Allow microphone access by starting a recording once, then try again.",
                    )
                }
                return
            }
        }
        stopVoiceSample.set(false)
        presentation.update {
            it.copy(voiceSampleRecording = com.example.voicememo.ui.VoiceSampleRecordingUi(isRecording = true))
        }
        viewModelScope.launch {
            val profiles = container.speakerProfiles
            val file = profiles.newSampleFile()
            // Transcription and speaker analysis pause while the microphone is in use.
            InferenceGate.recordingStarted(VOICE_SAMPLE_GATE)
            val message = try {
                val recording = recorder.record(file, stopVoiceSample::get) { elapsed ->
                    presentation.update {
                        it.copy(
                            voiceSampleRecording = it.voiceSampleRecording.copy(
                                elapsedSeconds = (elapsed / 1_000).toInt(),
                            ),
                        )
                    }
                }
                InferenceGate.recordingStopped(VOICE_SAMPLE_GATE)
                presentation.update {
                    it.copy(
                        voiceSampleRecording =
                            com.example.voicememo.ui.VoiceSampleRecordingUi(isProcessing = true),
                    )
                }
                when {
                    recording == null -> "The microphone could not be used."
                    profiles.addRecordedSample(profiles.self().id, file, recording.inputChannel)
                        is SpeakerProfiles.SampleResult.Added -> "Voice sample added."
                    else -> "Not enough clear speech. Speak for at least 10 seconds and try again."
                }
            } catch (failure: Exception) {
                if (failure is kotlinx.coroutines.CancellationException) throw failure
                file.delete()
                "The voice sample could not be saved."
            } finally {
                InferenceGate.recordingStopped(VOICE_SAMPLE_GATE)
                presentation.update {
                    it.copy(voiceSampleRecording = com.example.voicememo.ui.VoiceSampleRecordingUi())
                }
            }
            mutableEffects.emitMessage(message)
        }
    }

    private fun MemoDetail.generatedSegments(): List<TranscriptSegmentUi> =
        TranscriptSegments.generated(this, transcriptTokenizer).map { it.toUi() }

    private fun MemoDetail.isPlainTextEdit(): Boolean = TranscriptSegments.isPlainTextEdit(this)

    /**
     * Generated segments with the user's corrections applied. Plain-text edits
     * from before segment editing can't be aligned, so they have no segments.
     */
    private fun MemoDetail.timedSegments(): List<TranscriptSegmentUi> =
        TranscriptSegments.timed(this, transcriptTokenizer).map { it.toUi() }

    private fun TimedTranscriptSegment.toUi(): TranscriptSegmentUi =
        TranscriptSegmentUi(
            id = id,
            startMillis = startMillis,
            endMillis = endMillis,
            startLabel = formatDuration(startMillis),
            endLabel = formatDuration(endMillis),
            text = text,
            isEdited = isEdited,
        )

    private fun transcriptLanguageLabel(override: String?, detected: String?): String? {
        fun label(code: String) = defaultLanguages.firstOrNull { it.code == code }?.label
            ?: Locale(code).getDisplayLanguage(Locale.ENGLISH).ifBlank { code.uppercase() }
        return when {
            override != null -> label(override)
            detected != null -> "${label(detected)} · auto"
            else -> null
        }
    }

    private fun formatDuration(millis: Long): String {
        val totalSeconds = millis.coerceAtLeast(0L) / 1_000L
        val hours = totalSeconds / 3_600L
        val minutes = totalSeconds / 60L % 60L
        val seconds = totalSeconds % 60L
        return if (hours > 0) {
            "%d:%02d:%02d".format(hours, minutes, seconds)
        } else {
            "%02d:%02d".format(minutes, seconds)
        }
    }

    private fun formatDurationTenths(millis: Long): String {
        val totalTenths = millis.coerceAtLeast(0L) / 100L
        val minutes = totalTenths / 600L
        val seconds = totalTenths / 10L % 60L
        return "%02d:%02d.%d".format(minutes, seconds, totalTenths % 10L)
    }

    private fun parseDuration(label: String): Long {
        val values = label.split(':').mapNotNull { it.toLongOrNull() }
        return when (values.size) {
            2 -> (values[0] * 60L + values[1]) * 1_000L
            3 -> (values[0] * 3_600L + values[1] * 60L + values[2]) * 1_000L
            else -> 0L
        }
    }
}

internal fun TranscriptionJob?.isGeneratingRecordingMetadata(): Boolean =
    this?.status == TranscriptionJobStatus.RUNNING &&
        phase == com.example.voicememo.data.TranscriptionPhase.GENERATING_METADATA

internal fun transcriptionTimingLabel(
    enabled: Boolean,
    phase: TranscriptionPhase,
    transcriptionElapsedMillis: Long?,
    segmentInferenceMillis: List<Long>,
): String? {
    if (!enabled) return null
    val completedSegmentTimings = segmentInferenceMillis.filter { it >= 0L }
    return when (phase) {
        TranscriptionPhase.Transcribing,
        TranscriptionPhase.Finalizing,
        -> completedSegmentTimings
            .takeIf(List<Long>::isNotEmpty)
            ?.average()
            ?.let { average ->
                "Average per segment ${formatTimingMillis(average)}"
            }

        TranscriptionPhase.Complete -> {
            val elapsed = transcriptionElapsedMillis
                ?: completedSegmentTimings.sum().takeIf { completedSegmentTimings.isNotEmpty() }
            elapsed?.let { "Transcription time ${formatTimingMillis(it.toDouble())}" }
        }

        else -> null
    }
}

internal fun formatTimingMillis(millis: Double): String {
    val safeMillis = millis.coerceAtLeast(0.0)
    val totalSeconds = safeMillis / 1_000.0
    if (totalSeconds < 60.0) return String.format(Locale.US, "%.1f s", totalSeconds)
    val minutes = totalSeconds.toLong() / 60L
    val seconds = totalSeconds - minutes * 60L
    return String.format(Locale.US, "%dm %.1fs", minutes, seconds)
}

internal fun transcriptionThermalStatusLabel(
    enabled: Boolean,
    phase: TranscriptionPhase,
    maximumThermalStatus: Int?,
): String? {
    if (!enabled || maximumThermalStatus == null) return null
    val status = when (maximumThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "No throttling"
        PowerManager.THERMAL_STATUS_LIGHT -> "Light"
        PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
        else -> return null
    }
    return when (phase) {
        TranscriptionPhase.Transcribing,
        TranscriptionPhase.Finalizing,
        -> "Thermal status: $status"
        TranscriptionPhase.Complete -> "Peak thermal status: $status"
        else -> null
    }
}

/**
 * Real-time factor of the finished transcript.
 *
 * The job's wall-clock stopwatch only covers the segments the post-recording
 * worker still had to run: windows pre-transcribed while recording are
 * persisted before it starts, and it keeps ticking while the job is paused for
 * another recording. Every persisted chunk carries its own measured inference
 * time (pre-transcribed ones included), so their sum is the numerator whenever
 * it is available. The stopwatch is only a fallback for chunks stored before
 * inference timings were recorded.
 */
internal fun transcriptionRealTimeFactorLabel(
    enabled: Boolean,
    phase: TranscriptionPhase,
    transcriptionElapsedMillis: Long?,
    segmentInferenceMillis: List<Long>,
    audioDurationMillis: Long,
): String? {
    if (!enabled || phase != TranscriptionPhase.Complete || audioDurationMillis <= 0L) {
        return null
    }
    val completedSegmentTimings = segmentInferenceMillis.filter { it >= 0L }
    val processingMillis = completedSegmentTimings
        .takeIf(List<Long>::isNotEmpty)
        ?.sum()
        ?: transcriptionElapsedMillis
        ?: return null
    val factor = processingMillis.toDouble() / audioDurationMillis
    return String.format(Locale.US, "RTF %.3f", factor)
}

internal fun backupFileName(nowMillis: Long): String =
    "Voice-Memo-backup-" +
        SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date(nowMillis)) +
        ".zip"
