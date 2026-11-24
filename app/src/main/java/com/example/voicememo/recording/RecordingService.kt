package com.example.voicememo.recording

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.example.voicememo.MainActivity
import com.example.voicememo.R
import com.example.voicememo.VoiceMemoApplication
import com.example.voicememo.coordination.InferenceGate
import com.example.voicememo.transcription.ParakeetModelFiles
import com.example.voicememo.transcription.TranscriptionModel
import com.example.voicememo.transcription.windowConfig
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class RecordingService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val running = AtomicBoolean(false)
    private val paused = AtomicBoolean(false)
    private val stopRequested = AtomicBoolean(false)
    private val finalizing = AtomicBoolean(false)
    private var interruptionPause = false
    private var captureJob: Job? = null
    private var audioRecord: AudioRecord? = null
    private var writer: WavFileWriter? = null
    private var waveform: WaveformAccumulator? = null
    private var audioFile: File? = null
    private var waveformFile: File? = null
    private var memoDirectory: File? = null
    private var whisperWindowSpooler: WhisperWindowSpooler? = null
    private var preTranscriptionConfig: RecordingTranscriptionConfig? = null
    private var activeMemoId: String? = null
    private var createdAtEpochMillis = 0L
    private var sampleRate = PREFERRED_SAMPLE_RATE
    private var lastNotificationSecond = -1L
    private var audioFocusRequest: AudioFocusRequest? = null
    private var audioInputRoute: AudioInputRoute? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private val audioManager by lazy { getSystemService(AudioManager::class.java) }
    private val headsetAudioRouter by lazy { HeadsetAudioRouter(audioManager) }
    private val notificationManager by lazy { getSystemService(NotificationManager::class.java) }

    private val focusListener = AudioManager.OnAudioFocusChangeListener { change ->
        when (change) {
            AudioManager.AUDIOFOCUS_LOSS,
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT,
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK,
            -> serviceScope.launch { pauseCapture(interrupted = true) }

            AudioManager.AUDIOFOCUS_GAIN -> {
                if (interruptionPause) serviceScope.launch { resumeCapture() }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startCapture()
            ACTION_PAUSE -> serviceScope.launch { pauseCapture(interrupted = false) }
            ACTION_RESUME -> serviceScope.launch { resumeCapture() }
            ACTION_STOP -> stopCapture()
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopCapture()
        captureJob?.cancel()
        finalizeCapture(unexpectedError = "Recording service stopped unexpectedly")
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun startCapture() {
        if (running.get() || activeMemoId != null) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            RecordingRuntime.state.value = RecordingState.Failed(
                memoId = null,
                message = "Microphone permission is required to record.",
            )
            stopSelf()
            return
        }

        RecordingRuntime.state.value = RecordingState.Starting
        stopRequested.set(false)
        finalizing.set(false)
        val memoId = UUID.randomUUID().toString()
        activeMemoId = memoId
        createdAtEpochMillis = System.currentTimeMillis()

        val directory = File(filesDir, "memos/$memoId").apply { mkdirs() }
        memoDirectory = directory
        audioFile = File(directory, "audio.wav")
        waveformFile = File(directory, "waveform.vwf")

        startAsForeground(buildNotification(paused = false, elapsedMillis = 0L))
        InferenceGate.recordingStarted(memoId)

        captureJob = serviceScope.launch {
            try {
                if (!requestAudioFocus()) error("Another app is using audio.")
                audioInputRoute = headsetAudioRouter.prepare()
                val configured = createAudioRecord(audioInputRoute?.preferredInput)
                audioRecord = configured.first
                sampleRate = configured.second
                writer = WavFileWriter(requireNotNull(audioFile), sampleRate)
                waveform = WaveformAccumulator(sampleRate)
                val container = (application as VoiceMemoApplication).container
                if (container.transcriptionPreferences.transcribeWhileRecordingEnabled) {
                    val config = RecordingTranscriptionConfig(
                        timestampsEnabled =
                            container.transcriptionPreferences.timestampsEnabled,
                        voiceActivityDetectionEnabled =
                            container.transcriptionPreferences.voiceActivityDetectionEnabled,
                        gpuAccelerationEnabled =
                            container.transcriptionPreferences.gpuAccelerationEnabled,
                        transcriptionModelId = liveTranscriptionModel(
                            selected = TranscriptionModel.fromId(
                                container.transcriptionPreferences.transcriptionModelId,
                            ),
                            parakeetInstalled = ParakeetModelFiles.isInstalled(this@RecordingService),
                        ).id,
                    )
                    preTranscriptionConfig = config
                    whisperWindowSpooler = runCatching {
                        WhisperWindowSpooler(
                            memoDirectory = requireNotNull(memoDirectory),
                            sampleRate = sampleRate,
                            windowConfig = TranscriptionModel.fromId(
                                config.transcriptionModelId,
                            ).windowConfig(),
                        )
                    }.onFailure {
                        Log.e(TAG, "Could not prepare Whisper recording windows.", it)
                    }.getOrNull()
                }
                acquireWakeLock()

                running.set(!stopRequested.get())
                if (!running.get()) {
                    finalizeCapture()
                    return@launch
                }
                paused.set(false)
                configured.first.startRecording()
                if (whisperWindowSpooler != null) {
                    preTranscriptionConfig?.let { config ->
                        runCatching {
                            container.transcriptionCoordinator.enqueueWhileRecording(
                                memoId = memoId,
                                memoDirectory = requireNotNull(memoDirectory).absolutePath,
                                timestampsEnabled = config.timestampsEnabled,
                                voiceActivityDetectionEnabled =
                                    config.voiceActivityDetectionEnabled,
                                gpuAccelerationEnabled = config.gpuAccelerationEnabled,
                                transcriptionModelId = config.transcriptionModelId,
                            )
                        }.onFailure {
                            Log.e(TAG, "Could not queue Whisper during recording.", it)
                            runCatching { whisperWindowSpooler?.close() }
                            whisperWindowSpooler = null
                        }
                    }
                }
                Log.i(
                    TAG,
                    "Recording from input type=${configured.first.routedDevice?.type} " +
                        "at ${configured.second}Hz",
                )
                captureLoop(configured.first)
                finalizeCapture()
            } catch (cancelled: CancellationException) {
                finalizeCapture(unexpectedError = "Recording was interrupted.")
                throw cancelled
            } catch (error: Throwable) {
                finalizeCapture(
                    unexpectedError = error.message ?: "Recording failed unexpectedly.",
                )
            }
        }
    }

    private suspend fun captureLoop(recorder: AudioRecord) {
        val minimumBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val samples = ShortArray(maxOf(minimumBytes / 2, sampleRate / 10))

        while (serviceScope.isActive && running.get()) {
            if (paused.get()) {
                delay(40)
                continue
            }

            val read = recorder.read(samples, 0, samples.size, AudioRecord.READ_BLOCKING)
            when {
                read > 0 -> {
                    writer?.write(samples, read)
                    whisperWindowSpooler?.add(samples, read)
                    val buckets = waveform?.add(samples, read).orEmpty()
                    val elapsed = writer?.durationMillis ?: 0L
                    RecordingRuntime.state.value = RecordingState.Recording(
                        memoId = requireNotNull(activeMemoId),
                        elapsedMillis = elapsed,
                        waveform = buckets,
                    )
                    val second = elapsed / 1_000L
                    if (second != lastNotificationSecond) {
                        lastNotificationSecond = second
                        notificationManager.notify(
                            NOTIFICATION_ID,
                            buildNotification(paused = false, elapsedMillis = elapsed),
                        )
                    }
                }

                read == AudioRecord.ERROR_INVALID_OPERATION && paused.get() -> Unit
                read == AudioRecord.ERROR_DEAD_OBJECT -> error("The microphone became unavailable.")
                read < 0 -> error("Microphone read failed ($read).")
            }
        }
    }

    private suspend fun pauseCapture(interrupted: Boolean) {
        if (!running.get() || paused.getAndSet(true)) return
        interruptionPause = interrupted
        runCatching { audioRecord?.stop() }
        val current = RecordingRuntime.state.value
        val elapsed = when (current) {
            is RecordingState.Recording -> current.elapsedMillis
            is RecordingState.Paused -> current.elapsedMillis
            else -> writer?.durationMillis ?: 0L
        }
        val buckets = when (current) {
            is RecordingState.Recording -> current.waveform
            is RecordingState.Paused -> current.waveform
            else -> waveform?.snapshot().orEmpty()
        }
        RecordingRuntime.state.value = RecordingState.Paused(
            memoId = activeMemoId ?: return,
            elapsedMillis = elapsed,
            waveform = buckets,
            interrupted = interrupted,
        )
        notificationManager.notify(
            NOTIFICATION_ID,
            buildNotification(paused = true, elapsedMillis = elapsed),
        )
    }

    private suspend fun resumeCapture() {
        if (!running.get() || !paused.get()) return
        try {
            audioRecord?.startRecording()
            paused.set(false)
            interruptionPause = false
            val elapsed = writer?.durationMillis ?: 0L
            RecordingRuntime.state.value = RecordingState.Recording(
                memoId = activeMemoId ?: return,
                elapsedMillis = elapsed,
                waveform = waveform?.snapshot().orEmpty(),
            )
            notificationManager.notify(
                NOTIFICATION_ID,
                buildNotification(paused = false, elapsedMillis = elapsed),
            )
        } catch (error: Throwable) {
            finalizeCapture(error.message ?: "The microphone could not resume.")
        }
    }

    private fun stopCapture() {
        stopRequested.set(true)
        if (!running.getAndSet(false)) return
        RecordingRuntime.state.value = RecordingState.Finalizing(
            memoId = activeMemoId ?: return,
            elapsedMillis = writer?.durationMillis ?: 0L,
        )
        runCatching { audioRecord?.stop() }
    }

    private fun finalizeCapture(unexpectedError: String? = null) {
        if (!finalizing.compareAndSet(false, true)) return
        val memoId = activeMemoId ?: return
        activeMemoId = null
        running.set(false)
        paused.set(false)

        runCatching { audioRecord?.stop() }
        runCatching { audioRecord?.release() }
        audioRecord = null
        headsetAudioRouter.clear(audioInputRoute)
        audioInputRoute = null

        val currentWriter = writer
        val durationMillis = currentWriter?.durationMillis ?: 0L
        val finalizationError = runCatching { currentWriter?.finish() }.exceptionOrNull()
        writer = null

        val sidecarError = runCatching {
            waveform?.finishTo(requireNotNull(waveformFile))
        }.exceptionOrNull()
        waveform = null
        runCatching { whisperWindowSpooler?.close() }
            .onFailure { Log.e(TAG, "Could not finalize Whisper recording windows.", it) }
        whisperWindowSpooler = null

        abandonAudioFocus()
        releaseWakeLock()
        InferenceGate.recordingStopped(memoId)
        removeForegroundNotification()

        val wav = audioFile
        val sidecar = waveformFile
        val hasRecoverableRecording = durationMillis > 0L &&
            finalizationError == null &&
            sidecarError == null &&
            wav?.isFile == true &&
            (wav.length()) > 44L &&
            sidecar?.isFile == true
        val terminalState = if (hasRecoverableRecording) {
            RecordingState.Completed(
                memoId = memoId,
                audioPath = requireNotNull(wav).absolutePath,
                waveformPath = requireNotNull(sidecar).absolutePath,
                durationMillis = durationMillis,
                sampleRate = sampleRate,
                createdAtEpochMillis = createdAtEpochMillis,
                preTranscriptionConfig = preTranscriptionConfig,
            )
        } else {
            val finalError = finalizationError?.message
                ?: sidecarError?.message
                ?: unexpectedError
            RecordingState.Failed(
                memoId = memoId,
                message = finalError ?: "Recording could not be finalized.",
                recoverableAudioPath = wav?.takeIf { it.exists() && it.length() > 44L }?.absolutePath,
            )
        }
        audioFile = null
        waveformFile = null
        memoDirectory = null
        preTranscriptionConfig = null

        if (terminalState is RecordingState.Completed && application is RecordingCompletionSink) {
            serviceScope.launch {
                runCatching {
                    (application as RecordingCompletionSink).persistCompletedRecording(terminalState)
                }.onSuccess {
                    RecordingRuntime.state.value = terminalState
                }.onFailure { persistenceError ->
                    RecordingRuntime.state.value = RecordingState.Failed(
                        memoId = memoId,
                        message = persistenceError.message ?: "The recording could not be added to the library.",
                        recoverableAudioPath = terminalState.audioPath,
                    )
                }
                stopSelf()
            }
        } else {
            RecordingRuntime.state.value = terminalState
            stopSelf()
        }
    }

    @SuppressLint("MissingPermission")
    private fun createAudioRecord(
        preferredInput: android.media.AudioDeviceInfo?,
    ): Pair<AudioRecord, Int> {
        // startCapture performs a runtime permission check immediately before
        // launching the capture coroutine; construction is also inside its
        // SecurityException-safe failure boundary.
        var lastFailure: Throwable? = null
        for (rate in intArrayOf(PREFERRED_SAMPLE_RATE, FALLBACK_SAMPLE_RATE)) {
            val minimum = AudioRecord.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            if (minimum <= 0) continue
            try {
                val recorder = AudioRecord.Builder()
                    .setAudioSource(MediaRecorder.AudioSource.MIC)
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                            .build(),
                    )
                    .setBufferSizeInBytes(maxOf(minimum * 2, rate / 2))
                    .build()
                if (recorder.state == AudioRecord.STATE_INITIALIZED) {
                    if (preferredInput == null || recorder.setPreferredDevice(preferredInput)) {
                        return recorder to rate
                    }
                    Log.w(
                        TAG,
                        "Could not select preferred input type=${preferredInput.type}; " +
                            "using Android's active recording route.",
                    )
                    return recorder to rate
                }
                recorder.release()
            } catch (error: Throwable) {
                lastFailure = error
            }
        }
        throw IllegalStateException("No supported mono PCM16 microphone configuration.", lastFailure)
    }

    private fun requestAudioFocus(): Boolean {
        val result = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                )
                .setOnAudioFocusChangeListener(focusListener)
                .build()
            audioFocusRequest = request
            audioManager.requestAudioFocus(request)
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(
                focusListener,
                AudioManager.STREAM_MUSIC,
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT,
            )
        }
        return result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    private fun abandonAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            audioFocusRequest?.let(audioManager::abandonAudioFocusRequest)
            audioFocusRequest = null
        } else {
            @Suppress("DEPRECATION")
            audioManager.abandonAudioFocus(focusListener)
        }
    }

    private fun acquireWakeLock() {
        val manager = getSystemService(PowerManager::class.java)
        wakeLock = manager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$packageName:recording",
        ).apply { acquire() }
    }

    private fun releaseWakeLock() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
    }

    private fun startAsForeground(notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(paused: Boolean, elapsedMillis: Long): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val toggleAction = if (paused) ACTION_RESUME else ACTION_PAUSE
        val toggleTitle = if (paused) "Resume" else "Pause"
        val toggleIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, RecordingService::class.java).setAction(toggleAction),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this,
            2,
            Intent(this, RecordingService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle(if (paused) "Recording paused" else "Recording voice memo")
            .setContentText(formatDuration(elapsedMillis))
            .setContentIntent(openApp)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .addAction(0, toggleTitle, toggleIntent)
            .addAction(0, "Stop", stopIntent)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.recording_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = getString(R.string.recording_channel_description)
                setSound(null, null)
            },
        )
    }

    @Suppress("DEPRECATION")
    private fun removeForegroundNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            stopForeground(true)
        }
    }

    private fun formatDuration(millis: Long): String {
        val seconds = millis / 1_000L
        return "%02d:%02d".format(seconds / 60L, seconds % 60L)
    }

    companion object {
        const val ACTION_START = "com.example.voicememo.action.START_RECORDING"
        const val ACTION_PAUSE = "com.example.voicememo.action.PAUSE_RECORDING"
        const val ACTION_RESUME = "com.example.voicememo.action.RESUME_RECORDING"
        const val ACTION_STOP = "com.example.voicememo.action.STOP_RECORDING"

        private const val CHANNEL_ID = "voice_recording"
        private const val NOTIFICATION_ID = 1001
        private const val TAG = "RecordingService"
        private const val PREFERRED_SAMPLE_RATE = 16_000
        private const val FALLBACK_SAMPLE_RATE = 48_000
    }
}

/** Parakeet transcribes live only when its side-installed model is present. */
internal fun liveTranscriptionModel(
    selected: TranscriptionModel,
    parakeetInstalled: Boolean,
): TranscriptionModel =
    if (selected == TranscriptionModel.PARAKEET_TDT_0_6B_V3 && !parakeetInstalled) {
        TranscriptionModel.WHISPER_LARGE_V3_TURBO
    } else {
        selected
    }
