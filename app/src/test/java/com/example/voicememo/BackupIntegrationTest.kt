package com.example.voicememo

import java.util.Calendar
import org.junit.Assert.assertEquals
import org.junit.Test

class BackupIntegrationTest {
    @Test
    fun suggestedBackupNameIsStableAndFilesystemSafe() {
        val calendar = Calendar.getInstance().apply {
            set(2026, Calendar.JULY, 24, 12, 0, 0)
            set(Calendar.MILLISECOND, 0)
        }

        assertEquals(
            "Voice-Memo-backup-2026-07-24.zip",
            backupFileName(calendar.timeInMillis),
        )
    }
}
