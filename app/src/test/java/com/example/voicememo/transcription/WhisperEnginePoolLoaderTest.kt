package com.example.voicememo.transcription

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WhisperEnginePoolLoaderTest {
    @Test
    fun twoEnginesOpenConcurrentlyAndReturnInIndexOrder() {
        val entered = CountDownLatch(2)
        val maximumActive = AtomicInteger()
        val active = AtomicInteger()

        val result = WhisperEnginePoolLoader.open(2) { index ->
            val current = active.incrementAndGet()
            maximumActive.accumulateAndGet(current, ::maxOf)
            entered.countDown()
            assertTrue(entered.await(2, TimeUnit.SECONDS))
            active.decrementAndGet()
            FakeEngine(index)
        }

        assertEquals(2, maximumActive.get())
        assertEquals(listOf(0, 1), result.map(FakeEngine::index))
        result.forEach(FakeEngine::close)
    }

    @Test
    fun failureClosesEverySuccessfullyOpenedEngine() {
        val firstOpened = CountDownLatch(1)
        val mayFail = CountDownLatch(1)
        val first = FakeEngine(0)

        assertThrows(IllegalStateException::class.java) {
            WhisperEnginePoolLoader.open(2) { index ->
                if (index == 0) {
                    firstOpened.countDown()
                    mayFail.await(2, TimeUnit.SECONDS)
                    first
                } else {
                    firstOpened.await(2, TimeUnit.SECONDS)
                    mayFail.countDown()
                    throw IllegalStateException("boom")
                }
            }
        }

        assertTrue(first.closed)
    }

    private class FakeEngine(
        val index: Int,
    ) : AutoCloseable {
        var closed = false
            private set

        override fun close() {
            closed = true
        }
    }
}
