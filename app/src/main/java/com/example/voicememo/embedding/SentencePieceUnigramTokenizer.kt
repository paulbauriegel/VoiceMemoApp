package com.example.voicememo.embedding

import java.io.InputStream
import java.text.Normalizer

/**
 * Pure-Kotlin SentencePiece unigram tokenizer that produces XLM-RoBERTa ids,
 * as used by multilingual-e5-small.
 *
 * It reads the reviewed `sentencepiece.bpe.model` protobuf directly, so no
 * native tokenizer library is needed. Ids follow the fairseq layout of the
 * Hugging Face tokenizer: `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, and every
 * other SentencePiece id shifted by one. Normalization approximates
 * `nmt_nfkc` with Java NFKC plus whitespace and control-character cleanup;
 * `SentencePieceUnigramTokenizerTest` pins parity with the Hugging Face
 * tokenizer on de/en/es/fr text.
 */
class SentencePieceUnigramTokenizer private constructor(
    private val pieceIds: Map<String, Int>,
    private val pieceScores: FloatArray,
    private val maximumPieceLength: Int,
    private val unknownScore: Float,
) {
    /** Returns `<s> pieces… </s>`, truncated to [maximumLength] ids. */
    fun encode(text: String, maximumLength: Int = DEFAULT_MAXIMUM_LENGTH): LongArray {
        require(maximumLength >= 2) { "maximumLength must leave room for <s> and </s>." }
        val pieces = ArrayList<Int>()
        normalizedWords(text).forEach { word -> encodeWord(word, pieces) }
        val bodyLength = minOf(pieces.size, maximumLength - 2)
        val ids = LongArray(bodyLength + 2)
        ids[0] = BOS_ID.toLong()
        for (index in 0 until bodyLength) ids[index + 1] = pieces[index].toLong()
        ids[bodyLength + 1] = EOS_ID.toLong()
        return ids
    }

    /**
     * Splits normalized text into Metaspace words. Each word starts with the
     * `▁` marker, so pieces never cross a word boundary.
     */
    private fun normalizedWords(text: String): List<String> {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val cleaned = StringBuilder(normalized.length)
        var offset = 0
        while (offset < normalized.length) {
            val codePoint = normalized.codePointAt(offset)
            offset += Character.charCount(codePoint)
            // nmt_nfkc: tab, LF, FF and CR become spaces, other C0 controls
            // and DEL vanish, and zero-width/direction marks and a literal
            // `▁` act as spaces.
            // Other format characters (soft hyphen, word joiner) are kept.
            when {
                codePoint == 0x09 || codePoint == 0x0a || codePoint == 0x0c || codePoint == 0x0d ->
                    cleaned.append(' ')
                codePoint < 0x20 || codePoint == 0x7f -> Unit
                codePoint in SPACE_LIKE_FORMAT_CHARACTERS ||
                    Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint) ->
                    cleaned.append(' ')
                else -> cleaned.appendCodePoint(codePoint)
            }
        }
        // Like the Hugging Face pipeline: runs collapse, a leading space
        // vanishes unless it is all there is, and trailing whitespace survives
        // as a lone marker word.
        val collapsed = cleaned.replace(SPACE_RUN, " ")
        if (collapsed.isEmpty()) return emptyList()
        if (collapsed == " ") return listOf(WORD_MARKER)
        return collapsed.removePrefix(" ").split(' ').map { WORD_MARKER + it }
    }

    /** Viterbi segmentation that maximizes the summed piece log-probabilities. */
    private fun encodeWord(word: String, output: MutableList<Int>) {
        val length = word.length
        val bestScore = DoubleArray(length + 1) { Double.NEGATIVE_INFINITY }
        val bestStart = IntArray(length + 1) { -1 }
        val bestId = IntArray(length + 1) { UNKNOWN_ID }
        bestScore[0] = 0.0
        for (start in 0 until length) {
            if (bestScore[start] == Double.NEGATIVE_INFINITY) continue
            // Surrogate pairs are one character for SentencePiece.
            if (start > 0 && Character.isLowSurrogate(word[start]) &&
                Character.isHighSurrogate(word[start - 1])
            ) {
                continue
            }
            var matchedSingle = false
            val limit = minOf(length, start + maximumPieceLength)
            for (end in start + 1..limit) {
                val id = pieceIds[word.substring(start, end)] ?: continue
                val candidate = bestScore[start] + pieceScores[id]
                if (end == start + charLength(word, start)) matchedSingle = true
                if (candidate > bestScore[end]) {
                    bestScore[end] = candidate
                    bestStart[end] = start
                    bestId[end] = id
                }
            }
            if (!matchedSingle) {
                val end = start + charLength(word, start)
                val candidate = bestScore[start] + unknownScore
                if (candidate > bestScore[end]) {
                    bestScore[end] = candidate
                    bestStart[end] = start
                    bestId[end] = UNKNOWN_ID
                }
            }
        }
        val reversed = ArrayList<Int>()
        var end = length
        while (end > 0) {
            val id = bestId[end]
            // Consecutive unknown characters fuse into one <unk>, like the
            // Hugging Face unigram model.
            if (!(id == UNKNOWN_ID && reversed.lastOrNull() == UNKNOWN_ID)) reversed += id
            end = bestStart[end]
        }
        for (index in reversed.indices.reversed()) output += reversed[index]
    }

    private fun charLength(text: String, index: Int): Int =
        if (Character.isHighSurrogate(text[index]) && index + 1 < text.length &&
            Character.isLowSurrogate(text[index + 1])
        ) {
            2
        } else {
            1
        }

    companion object {
        const val DEFAULT_MAXIMUM_LENGTH = 256
        const val BOS_ID = 0
        const val PAD_ID = 1
        const val EOS_ID = 2
        const val UNKNOWN_ID = 3
        private const val WORD_MARKER = "▁"
        private val SPACE_RUN = Regex(" {2,}")
        private val SPACE_LIKE_FORMAT_CHARACTERS =
            setOf(0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0xfeff, 0x2581)
        private const val FAIRSEQ_OFFSET = 1
        private const val UNKNOWN_PENALTY = 10f

        // SentencePiece.Type values from sentencepiece_model.proto.
        private const val TYPE_NORMAL = 1
        private const val TYPE_USER_DEFINED = 4

        fun load(input: InputStream): SentencePieceUnigramTokenizer =
            parse(input.readBytes())

        /**
         * Parses `ModelProto.pieces` (field 1). Every other field is skipped;
         * normalization is fixed to nmt_nfkc, which the pinned asset uses.
         */
        fun parse(bytes: ByteArray): SentencePieceUnigramTokenizer {
            val reader = ProtoReader(bytes, 0, bytes.size)
            val pieces = ArrayList<String>(250_000)
            val scores = ArrayList<Float>(250_000)
            val types = ArrayList<Int>(250_000)
            while (reader.hasMore()) {
                val tag = reader.readVarint().toInt()
                if (tag ushr 3 == 1 && tag and 7 == 2) {
                    val length = reader.readVarint().toInt()
                    val piece = ProtoReader(bytes, reader.position, reader.position + length)
                    reader.position += length
                    var text = ""
                    var score = 0f
                    var type = TYPE_NORMAL
                    while (piece.hasMore()) {
                        val pieceTag = piece.readVarint().toInt()
                        when {
                            pieceTag == (1 shl 3 or 2) -> text = piece.readString()
                            pieceTag == (2 shl 3 or 5) -> score = piece.readFloat()
                            pieceTag == (3 shl 3) -> type = piece.readVarint().toInt()
                            else -> piece.skip(pieceTag and 7)
                        }
                    }
                    pieces += text
                    scores += score
                    types += type
                } else {
                    reader.skip(tag and 7)
                }
            }
            check(pieces.size > 3) { "SentencePiece model has no pieces." }

            // Fairseq ids: spm 0..2 are <unk>,<s>,</s>; the rest shift by one.
            val pieceIds = HashMap<String, Int>(pieces.size * 2)
            val pieceScores = FloatArray(pieces.size + FAIRSEQ_OFFSET)
            var maximumPieceLength = 1
            var minimumScore = Float.MAX_VALUE
            pieces.forEachIndexed { spmId, text ->
                val type = types[spmId]
                if (type != TYPE_NORMAL && type != TYPE_USER_DEFINED) return@forEachIndexed
                val id = spmId + FAIRSEQ_OFFSET
                pieceIds[text] = id
                pieceScores[id] = scores[spmId]
                maximumPieceLength = maxOf(maximumPieceLength, text.length)
                minimumScore = minOf(minimumScore, scores[spmId])
            }
            return SentencePieceUnigramTokenizer(
                pieceIds = pieceIds,
                pieceScores = pieceScores,
                maximumPieceLength = maximumPieceLength,
                unknownScore = minimumScore - UNKNOWN_PENALTY,
            )
        }
    }
}

private class ProtoReader(
    private val bytes: ByteArray,
    var position: Int,
    private val end: Int,
) {
    fun hasMore(): Boolean = position < end

    fun readVarint(): Long {
        var result = 0L
        var shift = 0
        while (true) {
            val byte = bytes[position++].toInt()
            result = result or ((byte and 0x7f).toLong() shl shift)
            if (byte and 0x80 == 0) return result
            shift += 7
            check(shift < 64) { "Malformed protobuf varint." }
        }
    }

    fun readString(): String {
        val length = readVarint().toInt()
        val value = String(bytes, position, length, Charsets.UTF_8)
        position += length
        return value
    }

    fun readFloat(): Float {
        val bits = (bytes[position].toInt() and 0xff) or
            ((bytes[position + 1].toInt() and 0xff) shl 8) or
            ((bytes[position + 2].toInt() and 0xff) shl 16) or
            ((bytes[position + 3].toInt() and 0xff) shl 24)
        position += 4
        return Float.fromBits(bits)
    }

    fun skip(wireType: Int) {
        when (wireType) {
            0 -> readVarint()
            1 -> position += 8
            2 -> {
                // Read the length first: `position += readVarint()` would use the stale position.
                val length = readVarint().toInt()
                position += length
            }
            5 -> position += 4
            else -> error("Unsupported protobuf wire type $wireType.")
        }
    }
}
