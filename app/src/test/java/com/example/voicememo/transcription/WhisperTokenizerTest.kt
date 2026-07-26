package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Test

class WhisperTokenizerTest {
    @Test
    fun byteLevelDecoder_restoresSpacesAndUtf8() {
        val tokenizer = WhisperTokenizer.fromTokens(
            mapOf(
                1 to "ĠHello",
                2 to "Ã©",
                50_257 to "<|endoftext|>",
            ),
            specialTokenIds = setOf(50_257),
        )

        assertEquals(" Helloé", tokenizer.decode(listOf(1, 2, 50_257)))
    }

    @Test
    fun lexicalLessThanToken_isNotMistakenForSpecialToken() {
        val tokenizer = WhisperTokenizer.fromTokens(mapOf(27 to "<"))

        assertEquals("<", tokenizer.decode(listOf(27)))
    }
}
