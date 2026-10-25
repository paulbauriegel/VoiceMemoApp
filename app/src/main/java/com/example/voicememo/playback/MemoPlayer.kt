package com.example.voicememo.playback

import android.content.Context
import androidx.core.net.toUri
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class PlaybackState(
    val memoId: String? = null,
    val isPlaying: Boolean = false,
    val positionMillis: Long = 0L,
    val durationMillis: Long = 0L,
    val error: String? = null,
)

interface MemoPlayer {
    val state: StateFlow<PlaybackState>

    fun playOrPause(memoId: String, audioPath: String)
    fun seekTo(positionMillis: Long)
    fun stop()
}

class Media3MemoPlayer(
    context: Context,
) : MemoPlayer, Player.Listener, Closeable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val player = ExoPlayer.Builder(context.applicationContext).build().also {
        it.addListener(this)
    }
    private val mutableState = MutableStateFlow(PlaybackState())
    private var ticker: Job? = null

    override val state: StateFlow<PlaybackState> = mutableState.asStateFlow()

    override fun playOrPause(memoId: String, audioPath: String) {
        if (mutableState.value.memoId == memoId) {
            if (player.isPlaying) player.pause() else player.play()
            publish()
            return
        }

        val file = File(audioPath)
        if (!file.isFile) {
            mutableState.value = PlaybackState(
                memoId = memoId,
                error = "The audio file is no longer available.",
            )
            return
        }
        player.setMediaItem(MediaItem.fromUri(file.toUri()))
        player.prepare()
        player.playWhenReady = true
        mutableState.value = PlaybackState(memoId = memoId)
        startTicker()
    }

    override fun seekTo(positionMillis: Long) {
        // Duration is unknown until the media is prepared; ExoPlayer applies
        // an early seek once it is.
        val duration = player.duration
        player.seekTo(
            if (duration > 0L) {
                positionMillis.coerceIn(0L, duration)
            } else {
                positionMillis.coerceAtLeast(0L)
            },
        )
        publish()
    }

    override fun stop() {
        player.stop()
        player.clearMediaItems()
        ticker?.cancel()
        mutableState.value = PlaybackState()
    }

    override fun onIsPlayingChanged(isPlaying: Boolean) {
        publish()
        if (isPlaying) startTicker()
    }

    override fun onPlaybackStateChanged(playbackState: Int) {
        if (playbackState == Player.STATE_ENDED) {
            player.seekTo(0L)
            player.pause()
        }
        publish()
    }

    override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
        mutableState.value = mutableState.value.copy(
            isPlaying = false,
            error = error.localizedMessage ?: "Playback failed.",
        )
    }

    private fun startTicker() {
        if (ticker?.isActive == true) return
        ticker = scope.launch {
            while (isActive) {
                publish()
                delay(250L)
            }
        }
    }

    private fun publish() {
        mutableState.value = mutableState.value.copy(
            isPlaying = player.isPlaying,
            positionMillis = player.currentPosition.coerceAtLeast(0L),
            durationMillis = player.duration.coerceAtLeast(0L),
            error = null,
        )
    }

    override fun close() {
        ticker?.cancel()
        player.removeListener(this)
        player.release()
        scope.cancel()
    }
}
