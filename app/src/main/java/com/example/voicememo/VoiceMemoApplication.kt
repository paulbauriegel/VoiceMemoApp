package com.example.voicememo

import android.app.Application
import com.example.voicememo.data.NewMemo
import com.example.voicememo.data.WaveformState
import com.example.voicememo.recording.RecordingCompletionSink
import com.example.voicememo.recording.RecordingState
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class VoiceMemoApplication : Application(), RecordingCompletionSink {
    private val containerDelegate = lazy { AppContainer(this) }
    val container: AppContainer by containerDelegate

    override suspend fun persistCompletedRecording(recording: RecordingState.Completed) {
        val title = SimpleDateFormat(
            "MMM d 'at' h:mm a",
            Locale.getDefault(),
        ).format(Date(recording.createdAtEpochMillis))
        container.memoRepository.createMemo(
            NewMemo(
                title = title,
                createdAtMillis = recording.createdAtEpochMillis,
                durationMillis = recording.durationMillis,
                audioPath = recording.audioPath,
                waveformPath = recording.waveformPath,
                waveformState = WaveformState.READY,
            ),
        )
    }

    override fun onTerminate() {
        if (containerDelegate.isInitialized()) containerDelegate.value.close()
        super.onTerminate()
    }
}
