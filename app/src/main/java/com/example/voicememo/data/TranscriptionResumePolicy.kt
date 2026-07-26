package com.example.voicememo.data

/**
 * Describes the atomic database changes required before a transcription job
 * may reuse durable chunk output.
 */
internal data class TranscriptionResumePreparation(
    val job: TranscriptionJob,
    val discardDerivedChunks: Boolean,
)

/**
 * Keeps resumable progress only when it was produced by the exact same
 * inference implementation. Legacy jobs have a null fingerprint and therefore
 * take the same safe reset path as a changed implementation.
 */
internal fun planTranscriptionResume(
    current: TranscriptionJob,
    inferenceFingerprint: String,
    updatedAtMillis: Long,
): TranscriptionResumePreparation {
    require(inferenceFingerprint.isNotBlank()) {
        "Inference fingerprint must not be blank."
    }
    if (current.inferenceFingerprint == inferenceFingerprint) {
        return TranscriptionResumePreparation(
            job = current,
            discardDerivedChunks = false,
        )
    }

    return TranscriptionResumePreparation(
        job = current.copy(
            detectedLanguage = null,
            completedChunks = 0,
            currentChunkIndex = null,
            currentToken = 0,
            warning = null,
            inferenceFingerprint = inferenceFingerprint,
            updatedAtMillis = updatedAtMillis,
        ),
        discardDerivedChunks = true,
    )
}
