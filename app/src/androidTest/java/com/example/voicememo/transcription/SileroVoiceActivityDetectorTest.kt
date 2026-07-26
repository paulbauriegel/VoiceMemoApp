package com.example.voicememo.transcription

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SileroVoiceActivityDetectorTest {
    @Test
    fun bundledModel_hasExpectedContractAndRejectsSilence() {
        SileroVoiceActivityDetector.open(ApplicationProvider.getApplicationContext()).use {
            assertFalse(
                it.hasSpeech(
                    audio16Khz = FloatArray(WhisperFeatureExtractor.SAMPLE_RATE),
                    shouldCancel = { false },
                ),
            )
        }
    }
}
