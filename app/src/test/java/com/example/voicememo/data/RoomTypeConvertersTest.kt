package com.example.voicememo.data

import org.junit.Assert.assertEquals
import org.junit.Test

class RoomTypeConvertersTest {
    private val converters = RoomTypeConverters()

    @Test
    fun tokenIdsRoundTripWithoutJsonAllocation() {
        val tokens = listOf(50258, 50259, 50360, 42, 50257)

        assertEquals(tokens, converters.stringToTokenIds(converters.tokenIdsToString(tokens)))
    }

    @Test
    fun emptyTokenListRoundTrips() {
        assertEquals(emptyList<Int>(), converters.stringToTokenIds(""))
    }
}
