package com.example.voicememo.importing

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.io.FileInputStream
import java.nio.ByteOrder
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

internal interface ContentAudioDecoder {
    suspend fun decode(
        source: File,
        sink: DecodedPcmSink,
        onProgress: (Float?) -> Unit,
    )
}

/**
 * Uses Android's bundled audio decoders. Consequently AAC/M4A, MP3, Ogg
 * Vorbis/Opus, FLAC and WAV support follows the codecs available on the device
 * without adding a native decoder or any networking dependency.
 */
internal class MediaCodecContentAudioDecoder : ContentAudioDecoder {
    override suspend fun decode(
        source: File,
        sink: DecodedPcmSink,
        onProgress: (Float?) -> Unit,
    ) {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            // The path overload may be resolved by a media service which
            // cannot traverse this app's private cache directory. Passing an
            // already-open descriptor keeps the source private.
            FileInputStream(source).use { input ->
                extractor.setDataSource(input.fd)
            }
            val trackIndex = findAudioTrack(extractor)
            require(trackIndex >= 0) { "The selected file has no audio track." }
            extractor.selectTrack(trackIndex)

            val trackFormat = extractor.getTrackFormat(trackIndex)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME)
                ?: error("The selected audio track has no codec type.")
            require(mime.startsWith("audio/")) { "The selected track is not audio." }
            val durationUs = trackFormat.longOrNull(MediaFormat.KEY_DURATION)
                ?.takeIf { it > 0L }

            // Android decoders default to PCM16, but explicitly requesting it
            // prevents float output on codecs which honor this optional key.
            trackFormat.setInteger(PCM_ENCODING_KEY, PCM_ENCODING_16_BIT)
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(trackFormat, null, null, 0)
            codec.start()

            decodeLoop(
                extractor = extractor,
                codec = codec,
                durationUs = durationUs,
                sink = sink,
                onProgress = onProgress,
            )
        } finally {
            codec?.let { activeCodec ->
                runCatching { activeCodec.stop() }
                runCatching { activeCodec.release() }
            }
            runCatching { extractor.release() }
        }
    }

    private suspend fun decodeLoop(
        extractor: MediaExtractor,
        codec: MediaCodec,
        durationUs: Long?,
        sink: DecodedPcmSink,
        onProgress: (Float?) -> Unit,
    ) {
        val info = MediaCodec.BufferInfo()
        val converter = PcmStreamConverter(sink)
        var inputEnded = false
        var outputEnded = false
        var outputFormat: PlatformPcmFormat? = null
        var lastReportedFraction = -1f
        onProgress(if (durationUs == null) null else 0f)

        while (!outputEnded) {
            currentCoroutineContext().ensureActive()

            if (!inputEnded) {
                val inputIndex = codec.dequeueInputBuffer(DEQUEUE_TIMEOUT_US)
                if (inputIndex >= 0) {
                    val input = requireNotNull(codec.getInputBuffer(inputIndex)).apply { clear() }
                    val size = extractor.readSampleData(input, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(
                            inputIndex,
                            0,
                            0,
                            0L,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                        )
                        inputEnded = true
                    } else {
                        codec.queueInputBuffer(
                            inputIndex,
                            0,
                            size,
                            extractor.sampleTime.coerceAtLeast(0L),
                            0,
                        )
                        extractor.advance()
                    }
                }
            }

            when (val outputIndex = codec.dequeueOutputBuffer(info, DEQUEUE_TIMEOUT_US)) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    outputFormat = codec.outputFormat.toPlatformPcmFormat()
                }

                else -> if (outputIndex >= 0) {
                    try {
                        if (
                            info.size > 0 &&
                            (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0
                        ) {
                            val format = outputFormat
                                ?: codec.outputFormat.toPlatformPcmFormat()
                                    .also { outputFormat = it }
                            val source = requireNotNull(codec.getOutputBuffer(outputIndex))
                                .duplicate()
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .apply {
                                    position(info.offset)
                                    limit(info.offset + info.size)
                                }
                            converter.consume(source, format)

                            val fraction = durationUs?.let { duration ->
                                (info.presentationTimeUs.toDouble() / duration)
                                    .toFloat()
                                    .coerceIn(0f, 1f)
                            }
                            if (
                                fraction == null ||
                                fraction - lastReportedFraction >= MIN_PROGRESS_STEP
                            ) {
                                onProgress(fraction)
                                if (fraction != null) lastReportedFraction = fraction
                            }
                        }
                        outputEnded =
                            (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                    } finally {
                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }
        }
        converter.finish()
        onProgress(1f)
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int {
        for (index in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)
            if (mime?.startsWith("audio/") == true) return index
        }
        return -1
    }

    private fun MediaFormat.toPlatformPcmFormat(): PlatformPcmFormat =
        PlatformPcmFormat(
            sampleRate = getInteger(MediaFormat.KEY_SAMPLE_RATE),
            channelCount = getInteger(MediaFormat.KEY_CHANNEL_COUNT),
            encoding = integerOrNull(PCM_ENCODING_KEY) ?: PCM_ENCODING_16_BIT,
        )

    private fun MediaFormat.integerOrNull(key: String): Int? =
        if (containsKey(key)) getInteger(key) else null

    private fun MediaFormat.longOrNull(key: String): Long? =
        if (containsKey(key)) getLong(key) else null

    private companion object {
        // MediaFormat.KEY_PCM_ENCODING was added after the app's min SDK, but
        // the underlying key is safely ignored by older platform codecs.
        const val PCM_ENCODING_KEY = "pcm-encoding"
        const val DEQUEUE_TIMEOUT_US = 10_000L
        const val MIN_PROGRESS_STEP = 0.01f
    }
}
