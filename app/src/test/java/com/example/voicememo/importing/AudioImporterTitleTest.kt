package com.example.voicememo.importing

import org.junit.Assert.assertEquals
import org.junit.Test

class AudioImporterTitleTest {
    @Test
    fun removesOnlyTheFinalFilenameExtension() {
        assertEquals("meeting.v2", titleFromDisplayName(" meeting.v2.m4a "))
        assertEquals("recording", titleFromDisplayName("recording.mp3"))
        assertEquals("recording", titleFromDisplayName("recording"))
        assertEquals(".memo", titleFromDisplayName(".memo"))
    }

    @Test
    fun missingNameUsesReadableFallback() {
        assertEquals("Imported voice memo", titleFromDisplayName(null))
        assertEquals("Imported voice memo", titleFromDisplayName("  "))
    }

    @Test
    fun controlCharactersAndRepeatedWhitespaceAreCollapsed() {
        assertEquals(
            "Team meeting notes",
            titleFromDisplayName(" Team\nmeeting\u0000  notes.m4a "),
        )
    }
}
