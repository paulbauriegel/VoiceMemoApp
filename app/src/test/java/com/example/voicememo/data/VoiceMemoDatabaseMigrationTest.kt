package com.example.voicememo.data

import androidx.sqlite.db.SupportSQLiteDatabase
import java.lang.reflect.Proxy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceMemoDatabaseMigrationTest {
    @Test
    fun migrationElevenToTwelveKeepsExistingTranscriptAndSeedsSelectedHistory() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase
        VoiceMemoDatabase.MIGRATION_11_12.migrate(database)
        val sql = executedSql.joinToString("\n").uppercase()
        assertTrue(sql.contains("CREATE TABLE IF NOT EXISTS TRANSCRIPT_VERSIONS"))
        assertTrue(sql.contains("INSERT INTO TRANSCRIPT_VERSIONS"))
        assertTrue(sql.contains("FROM TRANSCRIPTS T"))
        assertTrue(sql.contains("AND J.STATUS = 'SUCCEEDED'"))
        assertFalse(sql.contains("DROP "))
        assertFalse(sql.contains("DELETE FROM"))
    }

    @Test
    fun migrationThirteenToFourteenOnlyCreatesTheSchemaFourteenSpeakerTables() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase
        VoiceMemoDatabase.MIGRATION_13_14.migrate(database)

        // Every statement must be exactly what Room recorded for version 14.
        val schema = java.io.File("schemas/com.example.voicememo.data.VoiceMemoDatabase/14.json")
            .readText()
            .replace("`", "")
        val tables = executedSql.filter { it.startsWith("CREATE TABLE IF NOT EXISTS speaker_") }
        val indices = executedSql.filter { it.startsWith("CREATE INDEX IF NOT EXISTS index_speaker_") }
        assertEquals(executedSql.size, tables.size + indices.size)
        assertEquals(5, tables.size)
        assertEquals(5, Regex("\"tableName\": \"speaker_").findAll(schema).count())
        assertEquals(4, indices.size)
        for (statement in executedSql) {
            val table = if (statement.startsWith("CREATE TABLE")) {
                statement.substringAfter("EXISTS ").substringBefore(" (")
            } else {
                statement.substringAfter(" ON ").substringBefore(" (")
            }
            // Room keeps the table name inside index names and the name before " (".
            val recorded = statement.replace(" $table (", " \${TABLE_NAME} (")
            assertTrue(statement, schema.contains("\"createSql\": \"$recorded\""))
        }
    }

    @Test
    fun migrationTwelveToThirteenOnlyAddsNullableTimedWordsColumn() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase
        VoiceMemoDatabase.MIGRATION_12_13.migrate(database)
        assertEquals(
            listOf("ALTER TABLE transcript_chunks ADD COLUMN timedWordsJson TEXT"),
            executedSql,
        )
    }

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

    @Test
    fun migrationTenToElevenOnlyCreatesTheExportedEmbeddingTable() {
        val executedSql = mutableListOf<String>()
        val database = Proxy.newProxyInstance(
            SupportSQLiteDatabase::class.java.classLoader,
            arrayOf(SupportSQLiteDatabase::class.java),
        ) { _, method, arguments ->
            if (method.name == "execSQL") executedSql += arguments?.first() as String
            defaultValue(method.returnType)
        } as SupportSQLiteDatabase

        VoiceMemoDatabase.MIGRATION_10_11.migrate(database)

        val normalizedSql = executedSql.single().replace(Regex("\\s+"), " ").trim()
        assertFalse(normalizedSql.uppercase().contains("DROP "))
        assertFalse(normalizedSql.uppercase().contains("DELETE FROM"))
        // Gradle runs unit tests from the app module directory.
        val schema = java.io.File(
            "schemas/com.example.voicememo.data.VoiceMemoDatabase/11.json",
        ).readText()
        val exported = Regex(
            "\"createSql\": \"(CREATE TABLE IF NOT EXISTS `\\$\\{TABLE_NAME\\}` " +
                "\\(`memoId` TEXT NOT NULL, `segmentId` TEXT NOT NULL, `textSha256`[^\"]*)\"",
        ).find(schema)!!.groupValues[1]
            .replace("\\u0060", "`")
            .replace("\${TABLE_NAME}", "transcript_segment_embeddings")
        fun compact(sql: String) = sql.replace(Regex("\\s*([()])\\s*"), "$1")
        assertEquals(compact(exported), compact(normalizedSql))
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
