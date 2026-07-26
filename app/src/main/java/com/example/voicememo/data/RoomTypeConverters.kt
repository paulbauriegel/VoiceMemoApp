package com.example.voicememo.data

import androidx.room.TypeConverter

class RoomTypeConverters {
    @TypeConverter
    fun waveformStateToString(value: WaveformState): String = value.name

    @TypeConverter
    fun stringToWaveformState(value: String): WaveformState = WaveformState.valueOf(value)

    @TypeConverter
    fun jobStatusToString(value: TranscriptionJobStatus): String = value.name

    @TypeConverter
    fun stringToJobStatus(value: String): TranscriptionJobStatus =
        TranscriptionJobStatus.valueOf(value)

    @TypeConverter
    fun transcriptionPhaseToString(value: TranscriptionPhase): String = value.name

    @TypeConverter
    fun stringToTranscriptionPhase(value: String): TranscriptionPhase =
        TranscriptionPhase.valueOf(value)

    @TypeConverter
    fun chunkStatusToString(value: TranscriptChunkStatus): String = value.name

    @TypeConverter
    fun stringToChunkStatus(value: String): TranscriptChunkStatus =
        TranscriptChunkStatus.valueOf(value)

    @TypeConverter
    fun tokenIdsToString(value: List<Int>): String = value.joinToString(separator = ",")

    @TypeConverter
    fun stringToTokenIds(value: String): List<Int> {
        if (value.isBlank()) return emptyList()
        return value.split(',').map(String::toInt)
    }
}
