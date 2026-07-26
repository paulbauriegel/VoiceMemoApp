package com.example.voicememo.recording

import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.StateFlow

class AndroidRecorderController(
    context: Context,
) : RecorderController {
    private val applicationContext = context.applicationContext

    override val state: StateFlow<RecordingState> = RecordingRuntime.state

    override fun start() {
        ContextCompat.startForegroundService(
            applicationContext,
            Intent(applicationContext, RecordingService::class.java)
                .setAction(RecordingService.ACTION_START),
        )
    }

    override fun pause() = send(RecordingService.ACTION_PAUSE)

    override fun resume() = send(RecordingService.ACTION_RESUME)

    override fun stop() = send(RecordingService.ACTION_STOP)

    private fun send(action: String) {
        applicationContext.startService(
            Intent(applicationContext, RecordingService::class.java).setAction(action),
        )
    }
}
