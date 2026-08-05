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
        TranscriptChunk::class,
        TranscriptionJob::class,
        MemoSearchFts::class,
    ],
    version = 9,
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
                )
                .build()

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
    }
}
