package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Test

class ParakeetTokenizerTest {
    @Test
    fun decode_handlesMetaspaceUtf8FallbackAndSpecialTokens() {
        val tokenizer = ParakeetTokenizer.fromPieces(
            mapOf(
                0 to "<pad>",
                1 to "▁Hello",
                2 to "<0xC3>",
                3 to "<0xA9>",
                4 to "!",
            ),
        )

        assertEquals("Helloé!", tokenizer.decode(listOf(0, 1, 2, 3, 4)))
    }
}
