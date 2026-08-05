package com.example.voicememo.data

import androidx.sqlite.db.SupportSQLiteDatabase
import java.lang.reflect.Proxy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceMemoDatabaseMigrationTest {
    @Test
    fun migrationOneToTwoOnlyAddsNullableFingerprintColumn() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") {
                executedSql += arguments?.first() as String
            }
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_1_2.migrate(database)

        assertEquals(
            listOf(
                "ALTER TABLE transcription_jobs " +
                    "ADD COLUMN inferenceFingerprint TEXT",
            ),
            executedSql,
        )
        val normalizedSql = executedSql.single().uppercase()
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
        assertFalse(normalizedSql.contains("NOT NULL"))
    }

    @Test
    fun migrationTwoToThreeAddsNonNullWhisperDefaultWithoutDeletingData() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_2_3.migrate(database)

        assertEquals(
            listOf(
                "ALTER TABLE transcription_jobs " +
                    "ADD COLUMN transcriptionModelId TEXT NOT NULL " +
                    "DEFAULT 'whisper-large-v3-turbo'",
            ),
            executedSql,
        )
        val normalizedSql = executedSql.single().uppercase()
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
    }

    @Test
    fun migrationThreeToFourAddsProjectsWithoutChangingExistingMemoFiles() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_3_4.migrate(database)

        assertEquals(4, executedSql.size)
        val normalizedSql = executedSql.joinToString("\n").uppercase()
        assertTrue(normalizedSql.contains("CREATE TABLE IF NOT EXISTS `PROJECTS`"))
        assertTrue(normalizedSql.contains("ADD COLUMN `PROJECTID` TEXT"))
        assertTrue(normalizedSql.contains("ON DELETE SET NULL"))
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE FROM"))
    }

    @Test
    fun migrationFourToFiveAddsNonNullStarredDefaultWithoutDeletingData() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_4_5.migrate(database)

        assertEquals(
            listOf(
                "ALTER TABLE `memos` ADD COLUMN `isStarred` " +
                    "INTEGER NOT NULL DEFAULT 0",
            ),
            executedSql,
        )
        val normalizedSql = executedSql.single().uppercase()
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
    }

    @Test
    fun migrationFiveToSixAddsProjectIconAndArchiveDefaultsWithoutDeletingData() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_5_6.migrate(database)

        assertEquals(
            listOf(
                "ALTER TABLE `projects` ADD COLUMN `icon` TEXT NOT NULL DEFAULT 'WORK'",
                "ALTER TABLE `projects` ADD COLUMN `isArchived` INTEGER NOT NULL DEFAULT 0",
            ),
            executedSql,
        )
        val normalizedSql = executedSql.joinToString("\n").uppercase()
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
    }

    @Test
    fun migrationSixToSevenPreservesLegacyTitlesAndAddsGeneratedMetadata() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_6_7.migrate(database)

        assertEquals(4, executedSql.size)
        val normalizedSql = executedSql.joinToString("\n").uppercase()
        assertTrue(
            normalizedSql.contains(
                "ADD COLUMN `TITLEEDITEDBYUSER` INTEGER NOT NULL DEFAULT 1",
            ),
        )
        assertTrue(normalizedSql.contains("ADD COLUMN `GENERATEDSUMMARY` TEXT"))
        assertTrue(normalizedSql.contains("ADD COLUMN `METADATAGENERATEDATMILLIS` INTEGER"))
        assertTrue(normalizedSql.contains("ADD COLUMN `METADATAMODELID` TEXT"))
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
    }

    @Test
    fun migrationSevenToEightAddsNullableTimingWithoutDeletingData() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_7_8.migrate(database)

        assertEquals(2, executedSql.size)
        val normalizedSql = executedSql.joinToString("\n").uppercase()
        assertTrue(normalizedSql.contains("ADD COLUMN `TRANSCRIPTIONELAPSEDMILLIS` INTEGER"))
        assertTrue(normalizedSql.contains("ADD COLUMN `INFERENCEMILLIS` INTEGER"))
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
        assertFalse(normalizedSql.contains("NOT NULL"))
    }

    @Test
    fun migrationEightToNineAddsNullableThermalStatusWithoutDeletingData() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_8_9.migrate(database)

        assertEquals(
            listOf(
                "ALTER TABLE `transcription_jobs` " +
                    "ADD COLUMN `maximumThermalStatus` INTEGER",
            ),
            executedSql,
        )
        val normalizedSql = executedSql.single().uppercase()
        assertFalse(normalizedSql.contains("DROP "))
        assertFalse(normalizedSql.contains("DELETE "))
        assertFalse(normalizedSql.contains("NOT NULL"))
    }

    private fun defaultValue(type: Class<*>): Any? = when (type) {
        java.lang.Boolean.TYPE -> false
        java.lang.Byte.TYPE -> 0.toByte()
        java.lang.Short.TYPE -> 0.toShort()
        java.lang.Integer.TYPE -> 0
        java.lang.Long.TYPE -> 0L
        java.lang.Float.TYPE -> 0f
        java.lang.Double.TYPE -> 0.0
        java.lang.Character.TYPE -> '\u0000'
        else -> null
    }
}
