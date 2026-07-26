package com.example.voicememo.sharing

import org.junit.Assert.assertEquals
import org.junit.Test

class ShareExporterFileNameTest {
    @Test
    fun keepsUnicodeButRemovesPathSeparators() {
        assertEquals("Gespräch _ 23_07", safeShareFileName("  Gespräch / 23:07  "))
    }
}
