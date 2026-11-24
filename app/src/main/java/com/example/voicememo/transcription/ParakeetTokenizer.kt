package com.example.voicememo.transcription

import android.util.JsonReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

internal class ParakeetTokenizer private constructor(
    private val pieces: Array<String?>,
) {
    fun decode(tokenIds: List<Int>): String {
        val output = StringBuilder()
        val pendingBytes = mutableListOf<Byte>()

        fun flushBytes() {
            if (pendingBytes.isNotEmpty()) {
                output.append(pendingBytes.toByteArray().toString(StandardCharsets.UTF_8))
                pendingBytes.clear()
            }
        }

        tokenIds.forEach { tokenId ->
            val piece = pieces.getOrNull(tokenId) ?: return@forEach
            if (piece.startsWith("<") && piece.endsWith(">")) {
                val byteValue = BYTE_TOKEN.matchEntire(piece)
                    ?.groupValues
                    ?.get(1)
                    ?.toIntOrNull(16)
                if (byteValue != null) {
                    pendingBytes += byteValue.toByte()
                }
                return@forEach
            }
            flushBytes()
            output.append(piece.replace(METASPACE, " "))
        }
        flushBytes()
        return output.toString().trim()
    }

    /**
     * Groups timed tokens into display words: a piece starting with the SentencePiece
     * metaspace opens a new word; continuation and byte-fallback pieces join the current one.
     */
    fun words(tokens: List<ParakeetTimedToken>): List<WhisperTimedSegment> {
        val words = mutableListOf<WhisperTimedSegment>()
        var group = mutableListOf<ParakeetTimedToken>()
        fun flush() {
            if (group.isEmpty()) return
            val text = decode(group.map(ParakeetTimedToken::id))
            if (text.isNotEmpty()) {
                words += WhisperTimedSegment(group.first().startMillis, group.last().endMillis, text)
            }
            group = mutableListOf()
        }
        tokens.forEach { token ->
            if (pieces.getOrNull(token.id)?.startsWith(METASPACE) == true) flush()
            group += token
        }
        flush()
        return words
    }

    companion object {
        private const val METASPACE = "\u2581"
        private val BYTE_TOKEN = Regex("<0x([0-9A-Fa-f]{2})>")

        fun load(input: InputStream): ParakeetTokenizer {
            val piecesById = mutableMapOf<Int, String>()
            JsonReader(InputStreamReader(input, StandardCharsets.UTF_8)).use { reader ->
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "model" -> readModel(reader, piecesById)
                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
            }
            require(piecesById.isNotEmpty()) { "Parakeet tokenizer vocabulary is empty." }
            val pieces = arrayOfNulls<String>(piecesById.keys.max() + 1)
            piecesById.forEach { (id, piece) -> pieces[id] = piece }
            require(pieces.size == ParakeetAssets.BLANK_TOKEN_ID) {
                "Parakeet tokenizer has ${pieces.size} entries; expected 8192."
            }
            return ParakeetTokenizer(pieces)
        }

        internal fun fromPieces(pieces: Map<Int, String>): ParakeetTokenizer {
            val vocabulary = arrayOfNulls<String>(
                (pieces.keys.maxOrNull() ?: -1) + 1,
            )
            pieces.forEach { (id, piece) -> vocabulary[id] = piece }
            return ParakeetTokenizer(vocabulary)
        }

        private fun readModel(reader: JsonReader, piecesById: MutableMap<Int, String>) {
            reader.beginObject()
            while (reader.hasNext()) {
                when (reader.nextName()) {
                    "vocab" -> {
                        reader.beginObject()
                        while (reader.hasNext()) {
                            val piece = reader.nextName()
                            piecesById[reader.nextInt()] = piece
                        }
                        reader.endObject()
                    }
                    else -> reader.skipValue()
                }
            }
            reader.endObject()
        }
    }
}
