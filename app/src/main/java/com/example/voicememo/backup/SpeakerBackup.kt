package com.example.voicememo.backup

import androidx.room.withTransaction
import com.example.voicememo.data.SpeakerPerson
import com.example.voicememo.data.SpeakerVoiceSample
import com.example.voicememo.data.VoiceMemoDatabase
import java.io.File
import java.security.MessageDigest
import java.util.Properties
import java.util.UUID

/** The voice profile part of a backup: known people and their voice samples. */
internal class SpeakerBackup(
    val persons: List<SpeakerPerson>,
    val samples: List<SpeakerVoiceSample>,
)

internal interface SpeakerBackupStore {
    suspend fun snapshot(): SpeakerBackup

    /** The recorded audio of a sample, when it is still present. */
    fun sampleAudio(fileName: String): File?

    /**
     * Adds what is missing. The backup's "me" becomes this device's "me";
     * samples already present are skipped. [stagedAudio] maps a sample's audio
     * file name to its verified staged file, [memoIds] the backup's memo ids to
     * the restored ones. Returns the number of samples added.
     */
    suspend fun restore(backup: SpeakerBackup, stagedAudio: Map<String, File>, memoIds: Map<String, String>): Int
}

internal class RoomSpeakerBackupStore(
    private val database: VoiceMemoDatabase,
    private val sampleDirectory: File,
    private val onRestored: suspend () -> Unit = {},
) : SpeakerBackupStore {
    private val dao = database.voiceMemoDao()

    override suspend fun snapshot(): SpeakerBackup =
        database.withTransaction { SpeakerBackup(dao.getSpeakerPersons(), dao.getSpeakerVoiceSamples()) }

    override fun sampleAudio(fileName: String): File? =
        File(sampleDirectory, fileName).takeIf { it.parentFile == sampleDirectory && it.isFile }

    override suspend fun restore(
        backup: SpeakerBackup,
        stagedAudio: Map<String, File>,
        memoIds: Map<String, String>,
    ): Int {
        sampleDirectory.mkdirs()
        val added = database.withTransaction {
            val existingPersons = dao.getSpeakerPersons()
            val existingSelf = existingPersons.firstOrNull { it.isSelf }
            val personIds = HashMap<String, String>()
            for (person in backup.persons) {
                val target = when {
                    person.isSelf && existingSelf != null -> existingSelf.id
                    existingPersons.any { it.id == person.id } -> person.id
                    else -> person.id.also { dao.upsertSpeakerPerson(person) }
                }
                personIds[person.id] = target
            }
            val existingSamples = dao.getSpeakerVoiceSamples().mapTo(hashSetOf()) { it.id }
            var count = 0
            for (sample in backup.samples) {
                val personId = personIds[sample.personId] ?: continue
                if (!existingSamples.add(sample.id)) continue
                val audio = sample.audioFileName?.let { name ->
                    val staged = stagedAudio[name] ?: return@let null
                    val target = File(sampleDirectory, name)
                    if (target.exists() || staged.renameTo(target) || runCatching { staged.copyTo(target) }.isSuccess) {
                        name
                    } else {
                        null
                    }
                }
                dao.upsertSpeakerVoiceSample(
                    SpeakerVoiceSample(
                        id = sample.id,
                        personId = personId,
                        modelId = sample.modelId,
                        vector = sample.vector,
                        origin = sample.origin,
                        memoId = sample.memoId?.let(memoIds::get),
                        startMillis = sample.startMillis,
                        endMillis = sample.endMillis,
                        inputChannel = sample.inputChannel,
                        audioFileName = audio,
                        speechMillis = sample.speechMillis,
                        createdAtMillis = sample.createdAtMillis,
                    ),
                )
                count++
            }
            count
        }
        if (added > 0) onRestored()
        return added
    }
}

/**
 * `speakers/speakers.properties`: every value is validated on the way in, like
 * the rest of the backup, because the archive is untrusted input.
 */
internal object SpeakerBackupCodec {
    const val ENTRY = "speakers/speakers.properties"
    const val SAMPLE_DIRECTORY = "speakers/samples"
    const val MAX_PROPERTIES_BYTES = 8L * 1024L * 1024L
    const val MAX_SAMPLE_AUDIO_BYTES = 32L * 1024L * 1024L
    val SAMPLE_ENTRY = Regex("speakers/samples/([0-9A-Fa-f-]{36})\\.wav")
    private const val MAX_PERSONS = 200
    private const val MAX_SAMPLES = 2_000
    private const val MAX_NAME_CHARACTERS = 80
    private const val MAX_TEXT_CHARACTERS = 120
    private const val MAX_VECTOR_BYTES = 8 * 1024
    private val ORIGINS = setOf(SpeakerVoiceSample.ORIGIN_RECORDED, SpeakerVoiceSample.ORIGIN_SEGMENT)

    /** Audio digests are keyed by file name so the importer can verify each staged file. */
    class Decoded(val backup: SpeakerBackup, val audio: Map<String, Pair<Long, String>>)

    fun encode(backup: SpeakerBackup, audio: Map<String, File>): Properties = Properties().apply {
        setProperty("personCount", backup.persons.size.toString())
        backup.persons.forEachIndexed { index, person ->
            setProperty("person.$index.id", person.id)
            setProperty("person.$index.name", person.name)
            setProperty("person.$index.isSelf", person.isSelf.toString())
            setProperty("person.$index.createdAtMillis", person.createdAtMillis.toString())
        }
        setProperty("sampleCount", backup.samples.size.toString())
        backup.samples.forEachIndexed { index, sample ->
            val key = "sample.$index"
            setProperty("$key.id", sample.id)
            setProperty("$key.personId", sample.personId)
            setProperty("$key.modelId", sample.modelId)
            setProperty("$key.vector", sample.vector.joinToString("") { "%02x".format(it) })
            setProperty("$key.origin", sample.origin)
            setProperty("$key.speechMillis", sample.speechMillis.toString())
            setProperty("$key.createdAtMillis", sample.createdAtMillis.toString())
            sample.memoId?.let { setProperty("$key.memoId", it) }
            sample.startMillis?.let { setProperty("$key.startMillis", it.toString()) }
            sample.endMillis?.let { setProperty("$key.endMillis", it.toString()) }
            sample.inputChannel?.let { setProperty("$key.inputChannel", it) }
            val file = sample.audioFileName?.let(audio::get)
            if (file != null) {
                setProperty("$key.audio", file.name)
                setProperty("$key.audioBytes", file.length().toString())
                setProperty("$key.audioSha256", sha256(file))
            }
        }
    }

    fun decode(properties: Properties): Decoded {
        fun text(key: String, maximum: Int = MAX_TEXT_CHARACTERS): String =
            requireNotNull(properties.getProperty(key)) { "The backed-up voice profile is incomplete." }
                .also { require(it.length <= maximum) { "The backed-up voice profile has an invalid value." } }
        fun optional(key: String): String? = properties.getProperty(key)?.also {
            require(it.length <= MAX_TEXT_CHARACTERS) { "The backed-up voice profile has an invalid value." }
        }
        fun long(key: String): Long =
            requireNotNull(text(key).toLongOrNull()?.takeIf { it >= 0 }) { "The backed-up voice profile has an invalid number." }
        fun uuid(value: String): String = value.also {
            require(runCatching { UUID.fromString(it) }.isSuccess && it.length == 36) {
                "The backed-up voice profile has an invalid id."
            }
        }

        val personCount = long("personCount").toInt()
        require(personCount in 0..MAX_PERSONS) { "The backed-up voice profile lists too many people." }
        val persons = List(personCount) { index ->
            SpeakerPerson(
                id = uuid(text("person.$index.id")),
                name = text("person.$index.name", MAX_NAME_CHARACTERS).ifBlank { "Person" },
                isSelf = text("person.$index.isSelf").toBooleanStrict(),
                createdAtMillis = long("person.$index.createdAtMillis"),
            )
        }
        require(persons.map { it.id }.toSet().size == persons.size && persons.count { it.isSelf } <= 1) {
            "The backed-up voice profile lists a person more than once."
        }
        val sampleCount = long("sampleCount").toInt()
        require(sampleCount in 0..MAX_SAMPLES) { "The backed-up voice profile has too many samples." }
        val audio = HashMap<String, Pair<Long, String>>()
        val samples = List(sampleCount) { index ->
            val key = "sample.$index"
            val hex = text("$key.vector", MAX_VECTOR_BYTES * 2)
            require(hex.length % 8 == 0 && hex.isNotEmpty() && hex.all { it in "0123456789abcdef" }) {
                "The backed-up voice profile has an invalid voiceprint."
            }
            val audioName = optional("$key.audio")?.also { name ->
                require(SAMPLE_ENTRY.matches("$SAMPLE_DIRECTORY/$name")) { "The backed-up voice sample has an invalid name." }
                val digest = text("$key.audioSha256")
                require(digest.length == 64 && digest.all { it in "0123456789abcdef" }) {
                    "The backed-up voice sample has an invalid checksum."
                }
                require(audio.put(name, long("$key.audioBytes") to digest) == null) {
                    "The backed-up voice profile lists a sample file twice."
                }
            }
            SpeakerVoiceSample(
                id = uuid(text("$key.id")),
                personId = uuid(text("$key.personId")).also { id ->
                    require(persons.any { it.id == id }) { "A backed-up voice sample belongs to nobody." }
                },
                modelId = text("$key.modelId"),
                vector = ByteArray(hex.length / 2) { hex.substring(it * 2, it * 2 + 2).toInt(16).toByte() },
                origin = text("$key.origin").also { require(it in ORIGINS) { "A backed-up voice sample has an unknown origin." } },
                memoId = optional("$key.memoId")?.let(::uuid),
                startMillis = optional("$key.startMillis")?.let { long("$key.startMillis") },
                endMillis = optional("$key.endMillis")?.let { long("$key.endMillis") },
                inputChannel = optional("$key.inputChannel"),
                audioFileName = audioName,
                speechMillis = long("$key.speechMillis"),
                createdAtMillis = long("$key.createdAtMillis"),
            )
        }
        require(samples.map { it.id }.toSet().size == samples.size) { "The backed-up voice profile lists a sample twice." }
        return Decoded(SpeakerBackup(persons, samples), audio)
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
