package com.example.voicememo.transcription

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperEngineCacheTest {
    @Test
    fun activeAppReusesReleasedMatchingEngine() {
        val cache = WhisperEngineCache()
        cache.setAppActive(true)
        var opens = 0

        val first = cache.acquire(key()) { engineSet(FakeEngine().also { opens += 1 }) }
        val engine = first.engines.single()
        first.close()
        val second = cache.acquire(key()) { engineSet(FakeEngine().also { opens += 1 }) }

        assertSame(engine, second.engines.single())
        assertEquals(1, opens)
        second.close()
        cache.close()
    }

    @Test
    fun inactiveAppClosesEngineWhenLeaseFinishes() {
        val cache = WhisperEngineCache()
        val engine = FakeEngine()

        cache.acquire(key()) { engineSet(engine) }.close()

        assertTrue(engine.closed)
    }

    @Test
    fun leavingAppDefersEvictionUntilActiveLeaseFinishes() {
        val cache = WhisperEngineCache()
        cache.setAppActive(true)
        val engine = FakeEngine()
        val lease = cache.acquire(key()) { engineSet(engine) }

        cache.setAppActive(false)

        assertFalse(engine.closed)
        lease.close()
        assertTrue(engine.closed)
    }

    @Test
    fun newConfigurationEvictsPreviousIdleEngine() {
        val cache = WhisperEngineCache()
        cache.setAppActive(true)
        val oldEngine = FakeEngine()
        cache.acquire(key(timestampsEnabled = false)) { engineSet(oldEngine) }.close()

        val replacement = cache.acquire(key(timestampsEnabled = true)) {
            engineSet(FakeEngine())
        }

        assertTrue(oldEngine.closed)
        replacement.close()
        cache.close()
    }

    @Test
    fun memoryPressureEvictsIdleButNotLeasedEngine() {
        val cache = WhisperEngineCache()
        cache.setAppActive(true)
        val idle = FakeEngine()
        val idleLease = cache.acquire(key(timestampsEnabled = false)) { engineSet(idle) }
        val leased = FakeEngine()
        val lease = cache.acquire(key(timestampsEnabled = true)) { engineSet(leased) }
        idleLease.close()

        cache.evictIdle()

        assertTrue(idle.closed)
        assertFalse(leased.closed)
        lease.close()
        cache.close()
    }

    private fun key(timestampsEnabled: Boolean = false) = WhisperEngineCache.Key(
        model = TranscriptionModel.WHISPER_LARGE_V3_TURBO,
        requestedBackend = WhisperBackendRequest.CPU,
        workerCount = 1,
        threadsPerWorker = 3,
        timestampsEnabled = timestampsEnabled,
    )

    private fun engineSet(engine: WhisperEngine) = WhisperEngineCache.EngineSet(
        engines = listOf(engine),
        backendSelection = WhisperBackendSelection.cpuRequested(),
    )

    private class FakeEngine : WhisperEngine {
        var closed = false
            private set

        override fun transcribeChunk(
            audio: FloatArray,
            language: WhisperLanguage,
            shouldCancel: () -> Boolean,
            onTokenProgress: (generated: Int, limit: Int) -> Unit,
        ) = WhisperChunkResult(
            tokens = emptyList(),
            text = "",
            detectedLanguage = "und",
            inferenceMillis = 0,
            truncated = false,
        )

        override fun close() {
            closed = true
        }
    }
}
