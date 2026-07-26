package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MemoSearchQueryTest {
    @Test
    fun blankQueryUsesUnfilteredLibrary() {
        assertNull(MemoSearchQuery.toMatchExpression(" \t\n "))
    }

    @Test
    fun queryBuildsLiteralUnicodePrefixTerms() {
        assertEquals(
            "\"Grüße\"* AND \"café\"*",
            MemoSearchQuery.toMatchExpression("Grüße, café"),
        )
    }

    @Test
    fun ftsOperatorsCannotBeInjected() {
        assertEquals(
            "\"title\"* AND \"OR\"* AND \"transcript\"*",
            MemoSearchQuery.toMatchExpression("title OR transcript*"),
        )
    }

    @Test
    fun cjkTextRemainsSearchable() {
        assertEquals("\"こんにちは世界\"*", MemoSearchQuery.toMatchExpression("こんにちは世界"))
    }
}
