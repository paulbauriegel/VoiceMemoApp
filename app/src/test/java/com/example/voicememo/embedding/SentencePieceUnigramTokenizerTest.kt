package com.example.voicememo.embedding

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.BeforeClass
import org.junit.Test

class SentencePieceUnigramTokenizerTest {
    @Test
    fun matchesHuggingFaceTokenizerOnMultilingualText() {
        val fixture = javaClass.classLoader!!
            .getResourceAsStream("semantic-search/tokenizer-parity.tsv")!!
            .bufferedReader()
            .readLines()
            .filterNot { it.startsWith("#") || it.isBlank() }
        check(fixture.size > 50)
        fixture.forEach { line ->
            val (quoted, ids) = line.split('\t')
            val text = unquoteJson(quoted)
            val expected = ids.split(' ').map(String::toLong).toLongArray()
            assertArrayEquals("Token ids for $quoted", expected, tokenizer.encode(text))
        }
    }

    @Test
    fun truncationKeepsSpecialTokens() {
        val ids = tokenizer.encode("eins zwei drei vier fünf sechs sieben", maximumLength = 4)
        assertEquals(4, ids.size)
        assertEquals(SentencePieceUnigramTokenizer.BOS_ID.toLong(), ids.first())
        assertEquals(SentencePieceUnigramTokenizer.EOS_ID.toLong(), ids.last())
    }

    private fun unquoteJson(quoted: String): String {
        val body = quoted.removeSurrounding("\"")
        val result = StringBuilder(body.length)
        var index = 0
        while (index < body.length) {
            val char = body[index++]
            if (char != '\\') {
                result.append(char)
                continue
            }
            when (val escape = body[index++]) {
                'n' -> result.append('\n')
                't' -> result.append('\t')
                'r' -> result.append('\r')
                'b' -> result.append('\b')
                'f' -> result.append('\u000c')
                'u' -> {
                    result.append(body.substring(index, index + 4).toInt(16).toChar())
                    index += 4
                }
                else -> result.append(escape)
            }
        }
        return result.toString()
    }

    companion object {
        private lateinit var tokenizer: SentencePieceUnigramTokenizer

        @BeforeClass
        @JvmStatic
        fun loadTokenizer() {
            // Gradle runs unit tests from the app module directory.
            val model = File(
                "src/main/generatedEmbeddingAssets/" +
                    MultilingualE5Embedder.TOKENIZER_ASSET_PATH,
            )
            tokenizer = model.inputStream().use(SentencePieceUnigramTokenizer::load)
        }
    }
}
