package com.example.voicememo

import android.content.Context
import androidx.work.WorkManager
import com.example.voicememo.backup.RoomVoiceMemoBackupManager
import com.example.voicememo.backup.VoiceMemoBackupManager
import com.example.voicememo.data.MemoDeletionHooks
import com.example.voicememo.data.MemoRecord
import com.example.voicememo.data.MemoRepository
import com.example.voicememo.data.RoomMemoRepository
import com.example.voicememo.data.VoiceMemoDatabase
import com.example.voicememo.embedding.MultilingualE5Embedder
import com.example.voicememo.embedding.SemanticTranscriptIndex
import com.example.voicememo.importing.AndroidAudioImporter
import com.example.voicememo.importing.AudioImporter
import com.example.voicememo.playback.Media3MemoPlayer
import com.example.voicememo.playback.MemoPlayer
import com.example.voicememo.recording.AndroidRecorderController
import com.example.voicememo.recording.RecorderController
import com.example.voicememo.sharing.ShareExporter
import com.example.voicememo.settings.AppearancePreferences
import com.example.voicememo.settings.TranscriptionPreferences
import com.example.voicememo.transcription.WhisperEngineCache
import com.example.voicememo.work.TranscriptionCoordinator
import com.example.voicememo.work.WorkManagerTranscriptionCoordinator
import java.io.File
import java.io.Closeable
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import com.example.voicememo.speaker.PyannoteSegmenter
import com.example.voicememo.speaker.SpeakerLabelIndex
import com.example.voicememo.speaker.SpeakerProfiles
import com.example.voicememo.speaker.VoiceSampleRecorder
import com.example.voicememo.transcription.SileroVoiceActivityDetector
import com.example.voicememo.speaker.CamPlusPlusEmbedder
import kotlinx.coroutines.withContext

class AppContainer(
    context: Context,
) : Closeable {
    private val applicationContext = context.applicationContext
    val database: VoiceMemoDatabase by lazy { VoiceMemoDatabase.create(applicationContext) }
    val workManager: WorkManager by lazy { WorkManager.getInstance(applicationContext) }
    val recorderController: RecorderController by lazy { AndroidRecorderController(applicationContext) }
    val memoPlayer: MemoPlayer by lazy { Media3MemoPlayer(applicationContext) }
    val shareExporter: ShareExporter by lazy { ShareExporter(applicationContext) }
    val transcriptionPreferences: TranscriptionPreferences by lazy {
        TranscriptionPreferences(applicationContext)
    }
    internal val whisperEngineCache: WhisperEngineCache by lazy { WhisperEngineCache() }
    val appearancePreferences: AppearancePreferences by lazy {
        AppearancePreferences(applicationContext)
    }
    val backupManager: VoiceMemoBackupManager by lazy {
        RoomVoiceMemoBackupManager(
            database = database,
            memoRoot = File(applicationContext.filesDir, "memos"),
            speakerSampleDirectory = File(applicationContext.filesDir, "speaker-samples")
                .takeIf { BuildConfig.SPEAKER_LABELS_ENABLED },
            onSpeakersRestored = {
                // A backup may hold voice samples of an earlier voice model.
                runCatching { speakerProfiles.upgradeRecordedSamples() }
                speakerLabelIndex.relabelAll()
            },
        )
    }
    val transcriptionCoordinator: TranscriptionCoordinator by lazy {
        WorkManagerTranscriptionCoordinator(
            workManager = workManager,
            repository = memoRepository,
        )
    }

    val memoRepository: MemoRepository by lazy {
        RoomMemoRepository(
            database = database,
            deletionHooks = PrivateMemoDeletionHooks(
                context = applicationContext,
                workManager = workManager,
            ),
        )
    }
    val semanticTranscriptIndex: SemanticTranscriptIndex by lazy {
        SemanticTranscriptIndex(
            dao = database.voiceMemoDao(),
            openEmbedder = { MultilingualE5Embedder.open(applicationContext) },
        )
    }
    internal val speakerLabelIndex: SpeakerLabelIndex by lazy {
        SpeakerLabelIndex(
            dao = database.voiceMemoDao(),
            openSegmenter = { PyannoteSegmenter.open(applicationContext) },
            openEmbedder = { CamPlusPlusEmbedder.open(applicationContext) },
        )
    }
    internal val speakerProfiles: SpeakerProfiles by lazy {
        SpeakerProfiles(
            dao = database.voiceMemoDao(),
            index = speakerLabelIndex,
            sampleDirectory = File(applicationContext.filesDir, "speaker-samples"),
            openEmbedder = { CamPlusPlusEmbedder.open(applicationContext) },
            openVoiceActivityDetector = { SileroVoiceActivityDetector.open(applicationContext) },
        )
    }
    internal val voiceSampleRecorder: VoiceSampleRecorder by lazy { VoiceSampleRecorder(applicationContext) }
    val audioImporter: AudioImporter by lazy {
        AndroidAudioImporter(
            context = applicationContext,
            memoRepository = memoRepository,
        )
    }

    override fun close() {
        whisperEngineCache.close()
        (memoPlayer as? Closeable)?.close()
        if (database.isOpen) database.close()
    }
}

private class PrivateMemoDeletionHooks(
    context: Context,
    private val workManager: WorkManager,
) : MemoDeletionHooks {
    private val memoRoot = File(context.filesDir, "memos")

    override suspend fun cancelTranscriptions(memoIds: Set<String>) = withContext(Dispatchers.IO) {
        memoIds.forEach { memoId ->
            workManager.cancelAllWorkByTag("memo:$memoId").result.get()
        }
    }

    override suspend fun deletePrivateFiles(memos: List<MemoRecord>) = withContext(Dispatchers.IO) {
        val canonicalRoot = memoRoot.canonicalFile
        val directories = memos
            .flatMap { memo ->
                listOfNotNull(
                    File(memo.audioPath).parentFile,
                    memo.waveformPath?.let(::File)?.parentFile,
                )
            }
            .map(File::getCanonicalFile)
            .distinct()

        directories.forEach { directory ->
            require(directory.parentFile == canonicalRoot) {
                "Refusing to delete a path outside the private memo directory."
            }
        }

        val staged = mutableListOf<Pair<File, File>>()
        try {
            directories.filter(File::exists).forEach { source ->
                val target = File(
                    canonicalRoot,
                    ".deleting-${source.name}-${UUID.randomUUID()}",
                )
                check(source.renameTo(target)) {
                    "Could not stage ${source.name} for permanent deletion."
                }
                staged += source to target
            }
        } catch (failure: Throwable) {
            staged.asReversed().forEach { (source, target) ->
                if (target.exists()) target.renameTo(source)
            }
            throw failure
        }

        staged.forEach { (_, target) ->
            check(target.deleteRecursively()) {
                "Could not permanently delete ${target.name}."
            }
        }
    }
}
