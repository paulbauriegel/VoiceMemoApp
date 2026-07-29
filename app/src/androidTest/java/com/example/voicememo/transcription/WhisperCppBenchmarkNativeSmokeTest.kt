package com.example.voicememo.transcription

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WhisperCppBenchmarkNativeSmokeTest {
    @Test
    fun loadsPinnedCpuRuntime() {
        assumeTrue(isWhisperCppBenchmarkIncluded())

        WhisperCppBenchmarkNative.load()
        val runtimeInfo = WhisperCppBenchmarkNative.runtimeInfo()

        assertTrue(runtimeInfo.contains("whisper.cpp"))
        assertTrue(runtimeInfo.contains("f049fff95a089aa9969deb009cdd4892b3e74916"))
        assertTrue(runtimeInfo.contains("NEON"))
        assertTrue(runtimeInfo.contains("Vulkan="))
    }

    @Test
    fun loadsPinnedQ5ModelAndTranscribesSyntheticSilence() {
        assumeTrue(isWhisperCppBenchmarkIncluded())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelFile = File(
            context.filesDir,
            "whispercpp-models/${WhisperCppBenchmarkModel.Q5_0.fileName}",
        )
        assumeTrue(modelFile.isFile)

        WhisperCppBenchmarkEngine.open(modelFile, threadCount = 2).use { engine ->
            val result = engine.transcribeChunk(
                audio = FloatArray(16_000),
                language = WhisperLanguage.Forced("en"),
            )

            assertEquals("en", result.detectedLanguage)
            assertTrue(result.inferenceMillis > 0)
            assertTrue(result.warnings.single().startsWith("benchmark-only:backend=cpu:"))
        }
    }

    @Test
    fun loadsPinnedQ5ModelWithVulkanWithoutCpuFallback() {
        assumeTrue(isWhisperCppBenchmarkIncluded())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelFile = File(
            context.filesDir,
            "whispercpp-models/${WhisperCppBenchmarkModel.Q5_0.fileName}",
        )
        assumeTrue(modelFile.isFile)

        WhisperCppBenchmarkEngine.open(
            modelFile = modelFile,
            threadCount = 4,
            backend = WhisperCppBenchmarkBackend.VULKAN,
        ).use { engine ->
            val result = engine.transcribeChunk(
                audio = FloatArray(16_000),
                language = WhisperLanguage.Forced("en"),
            )

            assertEquals("en", result.detectedLanguage)
            assertTrue(result.inferenceMillis > 0)
            assertTrue(result.warnings.single().startsWith("benchmark-only:backend=vulkan:"))
        }
    }
}
