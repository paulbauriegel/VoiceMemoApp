package com.example.voicememo.transcription

import android.content.Context
import android.os.Debug
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Shared by the Parakeet runtime benchmarks (ggml, LiteRT): 30 s windows at a 25 s
 * stride, gap repair, staging helpers and the per-sample JSON report.
 *
 * Gap repair: a window can decode audible speech as blanks only. Token-free stretches of
 * at least two seconds with audible audio are re-decoded with a short window around the
 * gap (gap +/- 3 s, at most 30 s) and, in `full` mode only, with nudged full windows; the
 * retry with the most tokens inside the gap is spliced in. On a 54-minute memo
 * the nudged windows recovered almost nothing beyond the tight window at ~3x the decodes,
 * so `tight` is the default.
 */
internal object ParakeetBenchmarkSupport {
    const val SAMPLE_RATE = 16_000
    const val WINDOW_SECONDS = 30
    const val STRIDE_SECONDS = 25
    const val FRAME_SECONDS = 0.08
    private const val GAP_SECONDS = 2.0
    private const val LOUD_DBFS = -50.0
    private const val LOUD_FRACTION = 0.3
    private const val LEVEL_FRAME_SAMPLES = 1_600
    private const val TIGHT_PAD_SECONDS = 3.0
    private const val EDGE_SECONDS = 0.05
    private const val RECOVERED_TOKENS_PER_SECOND = 1.0
    private val NUDGE_SECONDS = doubleArrayOf(1.0, -1.5, 0.25)
    private val SAFE_FILE_NAME = Regex("[A-Za-z0-9._-]+")

    data class TimedPiece(val tokenId: Int, val text: String, val startSeconds: Double, val endSeconds: Double) {
        fun toJson(): JSONObject = JSONObject()
            .put("id", tokenId)
            .put("text", text)
            .put("start_s", startSeconds)
            .put("end_s", endSeconds)
    }

    class RepairStats {
        var decodes = 0
        var decodeMillis = 0L
        var gapsChecked = 0
        var gapsRepaired = 0
    }

    /** Decodes audio[start, end) and returns pieces with absolute times; must add to stats. */
    fun interface WindowDecoder {
        fun decode(audio: PcmAudio, start: Int, end: Int, stats: RepairStats): List<TimedPiece>
    }

    /** Transcribes one recording window by window and returns its report entry. */
    fun transcribeSample(
        name: String,
        audio: PcmAudio,
        decoder: WindowDecoder,
        gapRepair: Boolean,
        tightOnly: Boolean,
        onWindow: () -> Unit,
    ): JSONObject {
        val stats = RepairStats()
        val windows = JSONArray()
        val started = System.nanoTime()
        var start = 0
        while (true) {
            val end = min(start + WINDOW_SECONDS * SAMPLE_RATE, audio.size)
            val pieces = decodeRepaired(decoder, audio, start, end, gapRepair, tightOnly, stats)
            windows.put(
                JSONObject()
                    .put("start_sample", start)
                    .put("end_sample", end)
                    .put("pieces", JSONArray(pieces.map { it.toJson() })),
            )
            onWindow()
            if (end >= audio.size) break
            start += STRIDE_SECONDS * SAMPLE_RATE
        }
        val wallMillis = (System.nanoTime() - started) / 1_000_000
        val seconds = audio.size.toDouble() / SAMPLE_RATE
        return JSONObject()
            .put("audio", name)
            .put("audio_seconds", seconds)
            .put("wall_ms", wallMillis)
            .put("decode_ms", stats.decodeMillis)
            .put("decodes", stats.decodes)
            .put("gaps_checked", stats.gapsChecked)
            .put("gaps_repaired", stats.gapsRepaired)
            .put("rtf", wallMillis / 1000.0 / seconds)
            .put("windows", windows)
    }

    private fun decodeRepaired(
        decoder: WindowDecoder,
        audio: PcmAudio,
        start: Int,
        end: Int,
        gapRepair: Boolean,
        tightOnly: Boolean,
        stats: RepairStats,
    ): List<TimedPiece> {
        var pieces = decoder.decode(audio, start, end, stats)
        if (!gapRepair) return pieces
        val maximumSamples = WINDOW_SECONDS * SAMPLE_RATE
        for ((gapStart, gapEnd) in gaps(pieces, start.toDouble() / SAMPLE_RATE, end.toDouble() / SAMPLE_RATE)) {
            if (!audible(audio, gapStart, gapEnd)) continue
            stats.gapsChecked++
            val tries = buildList {
                val tightStart = max(0, ((gapStart - TIGHT_PAD_SECONDS) * SAMPLE_RATE).toInt())
                add(tightStart to min(audio.size, min(tightStart + maximumSamples, ((gapEnd + TIGHT_PAD_SECONDS) * SAMPLE_RATE).toInt())))
                if (!tightOnly) for (nudge in NUDGE_SECONDS) {
                    val nudged = max(0, start + (nudge * SAMPLE_RATE).toInt())
                    add(nudged to min(audio.size, nudged + (end - start)))
                }
            }
            var best = emptyList<TimedPiece>()
            for ((retryStart, retryEnd) in tries) {
                val candidate = decoder.decode(audio, retryStart, retryEnd, stats).filter {
                    it.startSeconds >= gapStart + EDGE_SECONDS && it.endSeconds <= gapEnd - EDGE_SECONDS
                }
                if (candidate.size > best.size) best = candidate
                if (candidate.size >= max(3.0, (gapEnd - gapStart) * RECOVERED_TOKENS_PER_SECOND)) break
            }
            if (best.isNotEmpty()) {
                stats.gapsRepaired++
                pieces = (pieces.filterNot { it.startSeconds > gapStart && it.startSeconds < gapEnd } + best)
                    .sortedBy { it.startSeconds }
            }
        }
        return pieces
    }

    /** Token-free stretches of at least [GAP_SECONDS] inside the window. */
    private fun gaps(pieces: List<TimedPiece>, windowStart: Double, windowEnd: Double): List<Pair<Double, Double>> {
        val result = mutableListOf<Pair<Double, Double>>()
        var cursor = windowStart
        for (piece in pieces.sortedBy { it.startSeconds }) {
            if (piece.startSeconds - cursor >= GAP_SECONDS) result += cursor to piece.startSeconds
            cursor = max(cursor, piece.endSeconds)
        }
        if (windowEnd - cursor >= GAP_SECONDS) result += cursor to windowEnd
        return result
    }

    private fun audible(audio: PcmAudio, startSeconds: Double, endSeconds: Double): Boolean {
        val from = (startSeconds * SAMPLE_RATE).toInt().coerceIn(0, audio.size)
        val to = (endSeconds * SAMPLE_RATE).toInt().coerceIn(from, audio.size)
        val frames = (to - from) / LEVEL_FRAME_SAMPLES
        if (frames == 0) return false
        var loud = 0
        for (frame in 0 until frames) {
            var sum = 0.0
            val base = from + frame * LEVEL_FRAME_SAMPLES
            for (index in base until base + LEVEL_FRAME_SAMPLES) sum += audio[index] * audio[index]
            if (20 * log10(sqrt(sum / LEVEL_FRAME_SAMPLES) + 1e-9) > LOUD_DBFS) loud++
        }
        return loud.toDouble() / frames >= LOUD_FRACTION
    }

    fun memory(): JSONObject {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        // File-backed pages (the mapped model) are evictable; graphics + native heap + private
        // other are what the process really pins, as in the shipped Whisper gate.
        val stats = info.memoryStats
        return JSONObject()
            .put("total_pss_kib", info.totalPss.toLong())
            .put("graphics_kib", stats["summary.graphics"]?.toLongOrNull() ?: -1L)
            .put("native_heap_kib", stats["summary.native-heap"]?.toLongOrNull() ?: -1L)
            .put("private_other_kib", stats["summary.private-other"]?.toLongOrNull() ?: -1L)
            .put("code_kib", stats["summary.code"]?.toLongOrNull() ?: -1L)
            .put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
    }

    fun stagedFile(context: Context, directory: String, name: String, extension: String): File {
        require(SAFE_FILE_NAME.matches(name) && name.endsWith(extension)) { "Invalid staged file name." }
        val parent = File(context.filesDir, directory).canonicalFile
        val file = File(parent, name).canonicalFile
        require(file.parentFile == parent && file.isFile) { "Staged file is missing: $directory/$name" }
        return file
    }

    /**
     * 16-bit PCM samples memory-mapped from the WAV file, so long recordings (an hour is
     * ~100 MB of PCM, ~200 MB as floats) do not have to fit in the Java heap. Windows are
     * converted to floats on demand.
     */
    class PcmAudio(private val samples: java.nio.ShortBuffer) {
        val size: Int get() = samples.limit()
        operator fun get(index: Int): Float = samples.get(index) / 32768f
        fun slice(start: Int, end: Int): FloatArray = FloatArray(end - start) { samples.get(start + it) / 32768f }
    }

    fun readPcm16Wav(file: File): PcmAudio = RandomAccessFile(file, "r").use { input ->
        val mapped = input.channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, input.length())
            .order(ByteOrder.LITTLE_ENDIAN)
        var position = 12
        var format: Triple<Int, Int, Int>? = null
        while (position + 8 <= mapped.limit()) {
            val id = String(ByteArray(4) { mapped.get(position + it) }, Charsets.US_ASCII)
            val size = mapped.getInt(position + 4)
            val body = position + 8
            if (id == "fmt ") {
                format = Triple(mapped.getShort(body + 2).toInt(), mapped.getInt(body + 4), mapped.getShort(body + 14).toInt())
            } else if (id == "data") {
                require(format == Triple(1, SAMPLE_RATE, 16)) { "Expected mono 16 kHz PCM16 WAV, got $format." }
                val data = mapped.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                data.position(body)
                data.limit(min(mapped.limit(), body + size))
                return@use PcmAudio(data.slice().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer())
            }
            position = body + size + (size and 1)
        }
        error("WAV has no data chunk.")
    }
}
