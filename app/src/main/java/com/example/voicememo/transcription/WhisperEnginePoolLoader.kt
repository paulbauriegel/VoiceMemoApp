package com.example.voicememo.transcription

import java.util.Collections
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Opens independent CPU interpreters concurrently so the second model arena
 * does not add its full initialization latency after the first one.
 */
internal object WhisperEnginePoolLoader {
    fun <T : AutoCloseable> open(
        count: Int,
        factory: (index: Int) -> T,
    ): List<T> {
        require(count > 0) { "At least one Whisper engine is required." }
        if (count == 1) return listOf(factory(0))

        val opened = Collections.synchronizedList(mutableListOf<T>())
        val executor = Executors.newFixedThreadPool(count) { runnable ->
            Thread(runnable, "Whisper-model-loader").apply { isDaemon = true }
        }
        val futures = (0 until count).map { index ->
            executor.submit(
                Callable {
                    factory(index).also(opened::add)
                },
            )
        }
        return try {
            futures.map { future ->
                try {
                    future.get()
                } catch (failure: ExecutionException) {
                    throw failure.cause ?: failure
                }
            }
        } catch (failure: Throwable) {
            futures.forEach { it.cancel(true) }
            executor.shutdownNow()
            runCatching { executor.awaitTermination(30, TimeUnit.SECONDS) }
            opened.toList().asReversed().forEach { engine ->
                runCatching { engine.close() }
            }
            throw failure
        } finally {
            executor.shutdown()
        }
    }
}
