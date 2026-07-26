package com.example.voicememo.transcription

import android.util.JsonReader
import android.util.JsonToken
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

class WhisperTokenizer internal constructor(
    private val idToToken: Map<Int, String>,
    private val specialTokenIds: Set<Int>,
) {
    fun decode(tokenIds: List<Int>): String {
        val bytes = ByteArrayOutputStream()
        for (tokenId in tokenIds) {
            if (tokenId in specialTokenIds) continue
            val token = idToToken[tokenId] ?: continue
            var characterIndex = 0
            while (characterIndex < token.length) {
                val codePoint = Character.codePointAt(token, characterIndex)
                val byteValue = BYTE_DECODER[codePoint]
                if (byteValue != null) {
                    bytes.write(byteValue)
                } else {
                    val original = String(Character.toChars(codePoint))
                        .toByteArray(StandardCharsets.UTF_8)
                    bytes.write(original)
                }
                characterIndex += Character.charCount(codePoint)
            }
        }
        return bytes.toByteArray().toString(StandardCharsets.UTF_8)
    }

    companion object {
        fun load(input: InputStream): WhisperTokenizer {
            val vocabulary = mutableMapOf<Int, String>()
            val special = mutableSetOf<Int>()
            JsonReader(InputStreamReader(input, StandardCharsets.UTF_8)).use { reader ->
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "model" -> readModel(reader, vocabulary)
                        "added_tokens" -> readAddedTokens(reader, vocabulary, special)
                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
            }
            require(vocabulary.isNotEmpty()) { "tokenizer.json contains no vocabulary." }
            return WhisperTokenizer(vocabulary, special)
        }

        internal fun fromTokens(
            tokens: Map<Int, String>,
            specialTokenIds: Set<Int> = emptySet(),
        ): WhisperTokenizer = WhisperTokenizer(tokens, specialTokenIds)

        private fun readModel(reader: JsonReader, vocabulary: MutableMap<Int, String>) {
            reader.beginObject()
            while (reader.hasNext()) {
                if (reader.nextName() == "vocab") {
                    reader.beginObject()
                    while (reader.hasNext()) {
                        val token = reader.nextName()
                        vocabulary[reader.nextInt()] = token
                    }
                    reader.endObject()
                } else {
                    reader.skipValue()
                }
            }
            reader.endObject()
        }

        private fun readAddedTokens(
            reader: JsonReader,
            vocabulary: MutableMap<Int, String>,
            special: MutableSet<Int>,
        ) {
            reader.beginArray()
            while (reader.hasNext()) {
                var id: Int? = null
                var content: String? = null
                var isSpecial = false
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "id" -> id = reader.nextInt()
                        "content" -> content = reader.nextString()
                        "special" -> isSpecial = reader.nextBoolean()
                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
                if (id != null && content != null) {
                    vocabulary[id] = content
                    if (isSpecial) special += id
                }
            }
            reader.endArray()
        }

        private val BYTE_DECODER: Map<Int, Int> = buildMap {
            val direct = buildList {
                addAll('!'.code..'~'.code)
                addAll(0xA1..0xAC)
                addAll(0xAE..0xFF)
            }
            direct.forEach { byte -> put(byte, byte) }
            var extra = 0
            for (byte in 0..255) {
                if (byte !in direct) {
                    put(256 + extra, byte)
                    extra++
                }
            }
        }
    }
}

internal data class WhisperGenerationConfig(
    val eosTokenId: Int,
    val decoderStartTokenId: Int,
    val transcribeTokenId: Int,
    val noTimestampsTokenId: Int,
    val languageToId: Map<String, Int>,
    val suppressTokens: Set<Int>,
    val beginSuppressTokens: Set<Int>,
    val maxInitialTimestampIndex: Int? = null,
) {
    val idToLanguage: Map<Int, String> = languageToId.entries.associate { (code, id) -> id to code }
    val timestampBeginTokenId: Int = noTimestampsTokenId + 1

    fun languageToken(code: String): Int =
        languageToId[code]
            ?: throw IllegalArgumentException("Whisper does not support language '$code'.")

    companion object {
        fun load(input: InputStream): WhisperGenerationConfig {
            var eosTokenId: Int? = null
            var decoderStartTokenId: Int? = null
            var noTimestampsTokenId: Int? = null
            var transcribeTokenId: Int? = null
            var maxInitialTimestampIndex: Int? = null
            val languages = linkedMapOf<String, Int>()
            val suppressed = linkedSetOf<Int>()
            val beginSuppressed = linkedSetOf<Int>()

            JsonReader(InputStreamReader(input, StandardCharsets.UTF_8)).use { reader ->
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "eos_token_id" -> eosTokenId = reader.nextInt()
                        "decoder_start_token_id" -> decoderStartTokenId = reader.nextInt()
                        "no_timestamps_token_id" -> noTimestampsTokenId = reader.nextInt()
                        "max_initial_timestamp_index" -> {
                            maxInitialTimestampIndex = if (reader.peek() == JsonToken.NULL) {
                                reader.nextNull()
                                null
                            } else {
                                reader.nextInt()
                            }
                        }
                        "lang_to_id" -> readLanguageMap(reader, languages)
                        "suppress_tokens" -> readIntegerArray(reader, suppressed)
                        "begin_suppress_tokens" -> readIntegerArray(reader, beginSuppressed)
                        "task_to_id" -> {
                            reader.beginObject()
                            while (reader.hasNext()) {
                                if (reader.nextName() == "transcribe") {
                                    transcribeTokenId = reader.nextInt()
                                } else {
                                    reader.skipValue()
                                }
                            }
                            reader.endObject()
                        }

                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
            }

            return WhisperGenerationConfig(
                eosTokenId = requireNotNull(eosTokenId) { "generation_config missing EOS." },
                decoderStartTokenId = requireNotNull(decoderStartTokenId) {
                    "generation_config missing decoder start token."
                },
                transcribeTokenId = requireNotNull(transcribeTokenId) {
                    "generation_config missing transcribe token."
                },
                noTimestampsTokenId = requireNotNull(noTimestampsTokenId) {
                    "generation_config missing no-timestamps token."
                },
                languageToId = languages,
                suppressTokens = suppressed,
                beginSuppressTokens = beginSuppressed,
                maxInitialTimestampIndex = maxInitialTimestampIndex,
            )
        }

        private fun readLanguageMap(reader: JsonReader, destination: MutableMap<String, Int>) {
            reader.beginObject()
            while (reader.hasNext()) {
                val wrappedCode = reader.nextName()
                val code = wrappedCode.removePrefix("<|").removeSuffix("|>")
                destination[code] = reader.nextInt()
            }
            reader.endObject()
        }

        private fun readIntegerArray(reader: JsonReader, destination: MutableSet<Int>) {
            reader.beginArray()
            while (reader.hasNext()) {
                if (reader.peek() == JsonToken.NUMBER) {
                    destination += reader.nextInt()
                } else {
                    reader.skipValue()
                }
            }
            reader.endArray()
        }
    }
}
