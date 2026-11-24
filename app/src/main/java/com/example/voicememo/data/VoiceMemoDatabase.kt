package com.example.voicememo.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        MemoRecord::class,
        ProjectRecord::class,
        TranscriptRecord::class,
        TranscriptVersion::class,
        TranscriptChunk::class,
        TranscriptionJob::class,
        MemoSearchFts::class,
        TranscriptSegmentEdit::class,
        TranscriptSegmentEmbedding::class,
        SpeakerPerson::class,
        SpeakerVoiceSample::class,
        SpeakerLocalVoiceprint::class,
        SpeakerAnalysis::class,
        SpeakerSegmentOverride::class,
    ],
    version = 14,
    exportSchema = true,
)
@TypeConverters(RoomTypeConverters::class)
abstract class VoiceMemoDatabase : RoomDatabase() {
    abstract fun voiceMemoDao(): VoiceMemoDao

    companion object {
        private const val DATABASE_NAME = "voice-memo.db"

        fun create(context: Context): VoiceMemoDatabase =
            Room.databaseBuilder(
                context.applicationContext,
                VoiceMemoDatabase::class.java,
                DATABASE_NAME,
            )
                .setJournalMode(JournalMode.WRITE_AHEAD_LOGGING)
                .addMigrations(
                    MIGRATION_1_2,
                    MIGRATION_2_3,
                    MIGRATION_3_4,
                    MIGRATION_4_5,
                    MIGRATION_5_6,
                    MIGRATION_6_7,
                    MIGRATION_7_8,
                    MIGRATION_8_9,
                    MIGRATION_9_10,
                    MIGRATION_10_11,
                    MIGRATION_11_12,
                    MIGRATION_12_13,
                    MIGRATION_13_14,
                )
                .build()

        /**
         * Speaker labels: persons and their voice samples, per-run voiceprints,
         * the resulting timeline and the user's own segment labels. Only adds
         * empty tables; existing data is untouched.
         */
        internal val MIGRATION_13_14 = object : Migration(13, 14) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS speaker_persons (id TEXT NOT NULL, name TEXT NOT NULL, " +
                    "isSelf INTEGER NOT NULL, createdAtMillis INTEGER NOT NULL, PRIMARY KEY(id))",
                )
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS speaker_voice_samples (id TEXT NOT NULL, " +
                    "personId TEXT NOT NULL, modelId TEXT NOT NULL, vector BLOB NOT NULL, " +
                    "origin TEXT NOT NULL, memoId TEXT, startMillis INTEGER, endMillis INTEGER, " +
                    "inputChannel TEXT, audioFileName TEXT, speechMillis INTEGER NOT NULL, " +
                    "createdAtMillis INTEGER NOT NULL, PRIMARY KEY(id), " +
                    "FOREIGN KEY(personId) REFERENCES speaker_persons(id) ON UPDATE NO ACTION ON DELETE CASCADE )",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_speaker_voice_samples_personId ON speaker_voice_samples (personId)",
                )
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS speaker_local_voiceprints (memoId TEXT NOT NULL, " +
                    "jobId TEXT NOT NULL, chunkIndex INTEGER NOT NULL, localSpeaker INTEGER NOT NULL, " +
                    "startSample INTEGER NOT NULL, activeFrames BLOB NOT NULL, vector BLOB NOT NULL, " +
                    "PRIMARY KEY(jobId, chunkIndex, localSpeaker), " +
                    "FOREIGN KEY(memoId) REFERENCES memos(id) ON UPDATE NO ACTION ON DELETE CASCADE )",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_speaker_local_voiceprints_memoId ON speaker_local_voiceprints (memoId)",
                )
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS speaker_analyses (jobId TEXT NOT NULL, " +
                    "memoId TEXT NOT NULL, modelId TEXT NOT NULL, sampleCount INTEGER NOT NULL, " +
                    "voiceprintCount INTEGER NOT NULL, analysedAtMillis INTEGER NOT NULL, timeline BLOB, " +
                    "profileKey TEXT, labelledAtMillis INTEGER, PRIMARY KEY(jobId), " +
                    "FOREIGN KEY(memoId) REFERENCES memos(id) ON UPDATE NO ACTION ON DELETE CASCADE )",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_speaker_analyses_memoId ON speaker_analyses (memoId)",
                )
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS speaker_segment_overrides (memoId TEXT NOT NULL, " +
                    "jobId TEXT NOT NULL, segmentId TEXT NOT NULL, label TEXT NOT NULL, " +
                    "updatedAtMillis INTEGER NOT NULL, PRIMARY KEY(jobId, segmentId), " +
                    "FOREIGN KEY(memoId) REFERENCES memos(id) ON UPDATE NO ACTION ON DELETE CASCADE )",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS index_speaker_segment_overrides_memoId ON speaker_segment_overrides (memoId)",
                )
            }
        }

        /** Word timestamps for Parakeet windows; existing rows keep NULL. */
        internal val MIGRATION_12_13 = object : Migration(12, 13) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE transcript_chunks ADD COLUMN timedWordsJson TEXT")
            }
        }

        internal val MIGRATION_11_12 = object : Migration(11, 12) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS transcript_versions (
                        id TEXT NOT NULL PRIMARY KEY, memoId TEXT NOT NULL,
                        jobId TEXT, modelId TEXT, selected INTEGER NOT NULL,
                        saved_memoId TEXT NOT NULL, saved_generatedText TEXT,
                        saved_editedText TEXT, saved_detectedLanguage TEXT,
                        saved_generatedAtMillis INTEGER, saved_editedAtMillis INTEGER,
                        saved_generatedRevision INTEGER NOT NULL, segmentEdits TEXT NOT NULL,
                        FOREIGN KEY(memoId) REFERENCES memos(id) ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS index_transcript_versions_memoId ON transcript_versions(memoId)")
                // Existing transcripts and corrections remain in place. Before a version
                // is switched or replaced the repository snapshots their current edits.
                db.execSQL("""
                    INSERT INTO transcript_versions
                    SELECT 'legacy:' || t.memoId, t.memoId,
                        (SELECT id FROM transcription_jobs j WHERE j.memoId = t.memoId
                         AND j.status = 'SUCCEEDED' ORDER BY updatedAtMillis DESC, createdAtMillis DESC, id DESC LIMIT 1),
                        (SELECT transcriptionModelId FROM transcription_jobs j WHERE j.memoId = t.memoId
                         AND j.status = 'SUCCEEDED' ORDER BY updatedAtMillis DESC, createdAtMillis DESC, id DESC LIMIT 1),
                        1, t.memoId, t.generatedText, t.editedText, t.detectedLanguage,
                        t.generatedAtMillis, t.editedAtMillis, t.generatedRevision, ''
                    FROM transcripts t
                """.trimIndent())
            }
        }

        internal val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE transcription_jobs " +
                        "ADD COLUMN inferenceFingerprint TEXT",
                )
            }
        }

        internal val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE transcription_jobs " +
                        "ADD COLUMN transcriptionModelId TEXT NOT NULL " +
                        "DEFAULT 'whisper-large-v3-turbo'",
                )
            }
        }

        internal val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `projects` (
                        `id` TEXT NOT NULL,
                        `name` TEXT NOT NULL,
                        `createdAtMillis` INTEGER NOT NULL,
                        `updatedAtMillis` INTEGER NOT NULL,
                        PRIMARY KEY(`id`)
                    )
                    """.trimIndent(),
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_projects_updatedAtMillis` " +
                        "ON `projects` (`updatedAtMillis`)",
                )
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `projectId` TEXT " +
                        "REFERENCES `projects`(`id`) ON UPDATE NO ACTION ON DELETE SET NULL",
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_memos_projectId` " +
                        "ON `memos` (`projectId`)",
                )
            }
        }

        internal val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `isStarred` " +
                        "INTEGER NOT NULL DEFAULT 0",
                )
            }
        }

        internal val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `projects` ADD COLUMN `icon` " +
                        "TEXT NOT NULL DEFAULT 'WORK'",
                )
                db.execSQL(
                    "ALTER TABLE `projects` ADD COLUMN `isArchived` " +
                        "INTEGER NOT NULL DEFAULT 0",
                )
            }
        }

        internal val MIGRATION_6_7 = object : Migration(6, 7) {
            override fun migrate(db: SupportSQLiteDatabase) {
                // Existing titles may have been edited, so migration defaults
                // fail closed and never lets the model overwrite them.
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `titleEditedByUser` " +
                        "INTEGER NOT NULL DEFAULT 1",
                )
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `generatedSummary` TEXT",
                )
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `metadataGeneratedAtMillis` INTEGER",
                )
                db.execSQL(
                    "ALTER TABLE `memos` ADD COLUMN `metadataModelId` TEXT",
                )
            }
        }

        internal val MIGRATION_7_8 = object : Migration(7, 8) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `transcription_jobs` " +
                        "ADD COLUMN `transcriptionElapsedMillis` INTEGER",
                )
                db.execSQL(
                    "ALTER TABLE `transcript_chunks` " +
                        "ADD COLUMN `inferenceMillis` INTEGER",
                )
            }
        }

        internal val MIGRATION_8_9 = object : Migration(8, 9) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `transcription_jobs` " +
                        "ADD COLUMN `maximumThermalStatus` INTEGER",
                )
            }
        }

        internal val MIGRATION_9_10 = object : Migration(9, 10) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `transcript_segment_edits` (
                        `memoId` TEXT NOT NULL,
                        `segmentId` TEXT NOT NULL,
                        `text` TEXT NOT NULL,
                        `editedAtMillis` INTEGER NOT NULL,
                        PRIMARY KEY(`memoId`, `segmentId`),
                        FOREIGN KEY(`memoId`) REFERENCES `memos`(`id`)
                            ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
            }
        }

        internal val MIGRATION_10_11 = object : Migration(10, 11) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `transcript_segment_embeddings` (
                        `memoId` TEXT NOT NULL,
                        `segmentId` TEXT NOT NULL,
                        `textSha256` TEXT NOT NULL,
                        `modelId` TEXT NOT NULL,
                        `vector` BLOB NOT NULL,
                        `embeddedAtMillis` INTEGER NOT NULL,
                        PRIMARY KEY(`memoId`, `segmentId`),
                        FOREIGN KEY(`memoId`) REFERENCES `memos`(`id`)
                            ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent(),
                )
            }
        }
    }
}
