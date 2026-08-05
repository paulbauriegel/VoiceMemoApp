package com.example.voicememo.transcription

import java.io.Closeable

/**
 * Retains the most recently used local ASR engine set while the app is visible.
 * Leased engines are never handed to two workers at once and are only evicted
 * after their current inference finishes.
 */
internal class WhisperEngineCache : Closeable {
    data class Key(
        val model: TranscriptionModel,
        val requestedBackend: WhisperBackendRequest,
        val workerCount: Int,
        val threadsPerWorker: Int,
        val timestampsEnabled: Boolean,
    )

    data class EngineSet(
        val engines: List<WhisperEngine>,
        val backendSelection: WhisperBackendSelection,
    ) {
        init {
            require(engines.isNotEmpty()) { "At least one Whisper engine is required." }
        }
    }

    internal data class Entry(
        val key: Key,
        val engineSet: EngineSet,
        var leased: Boolean,
    )

    class Lease internal constructor(
        private val cache: WhisperEngineCache,
        private val entry: Entry,
    ) : Closeable {
        val engines: List<WhisperEngine>
            get() = entry.engineSet.engines

        val backendSelection: WhisperBackendSelection
            get() = entry.engineSet.backendSelection

        override fun close() = cache.release(entry)
    }

    private val lock = Any()
    private val entries = mutableListOf<Entry>()
    private var appActive = false
    private var closed = false

    fun setAppActive(active: Boolean) {
        val evicted = synchronized(lock) {
            if (closed || appActive == active) return
            appActive = active
            if (active) emptyList() else removeIdleLocked()
        }
        closeEntries(evicted)
    }

    /** Releases cached engines without interrupting an engine currently in use. */
    fun evictIdle() {
        val evicted = synchronized(lock) {
            if (closed) return
            removeIdleLocked()
        }
        closeEntries(evicted)
    }

    fun acquire(key: Key, factory: () -> EngineSet): Lease {
        val cached = synchronized(lock) {
            check(!closed) { "Whisper engine cache is closed." }
            entries.firstOrNull { it.key == key && !it.leased }?.also { it.leased = true }
        }
        if (cached != null) return Lease(this, cached)

        val created = Entry(key = key, engineSet = factory(), leased = true)
        val stale = synchronized(lock) {
            if (closed) {
                listOf(created)
            } else {
                entries += created
                entries.filter { it !== created && !it.leased }.also(entries::removeAll)
            }
        }
        closeEntries(stale)
        check(created !in stale) { "Whisper engine cache was closed while loading." }
        return Lease(this, created)
    }

    private fun release(entry: Entry) {
        val evicted = synchronized(lock) {
            if (!entry.leased) return
            entry.leased = false
            if (!appActive || closed) {
                entries.remove(entry)
                listOf(entry)
            } else {
                entries.filter { it !== entry && !it.leased }.also(entries::removeAll)
            }
        }
        closeEntries(evicted)
    }

    override fun close() {
        val evicted = synchronized(lock) {
            if (closed) return
            closed = true
            entries.toList().also { entries.clear() }
        }
        closeEntries(evicted)
    }

    private fun removeIdleLocked(): List<Entry> =
        entries.filter { !it.leased }.also(entries::removeAll)

    private fun closeEntries(entries: List<Entry>) {
        entries.asReversed().forEach { entry ->
            entry.engineSet.engines.asReversed().forEach { engine ->
                runCatching { engine.close() }
            }
        }
    }
}
