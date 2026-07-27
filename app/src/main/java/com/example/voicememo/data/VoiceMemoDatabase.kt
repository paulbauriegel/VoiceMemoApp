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
        TranscriptRecord::class,
        TranscriptChunk::class,
        TranscriptionJob::class,
        MemoSearchFts::class,
    ],
    version = 3,
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
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
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
    }
}
