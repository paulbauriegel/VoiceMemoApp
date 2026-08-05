package com.example.voicememo.work

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TranscriptionThermalStatusTest {
    @Test
    fun maximumThermalStatusRetainsTheHighestAvailableSample() {
        assertEquals(3, maximumThermalStatus(previous = 3, current = 1))
        assertEquals(4, maximumThermalStatus(previous = 2, current = 4))
        assertEquals(2, maximumThermalStatus(previous = null, current = 2))
        assertEquals(2, maximumThermalStatus(previous = 2, current = null))
        assertNull(maximumThermalStatus(previous = null, current = null))
    }
}
