#include <jni.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "parakeet.h"

// Diagnostic-only JNI adapter for whisper.cpp's Parakeet TDT backend. One call
// transcribes one audio window and keeps token ids, encoder frames, durations
// and pieces for the Kotlin side to turn into timed words.

namespace {

struct ParakeetEngine {
    parakeet_context * context = nullptr;
    int threads = 1;
    std::mutex mutex;
    std::vector<int32_t> token_triplets;  // id, frame_index, duration_value
    std::vector<std::string> pieces;
    int64_t inference_millis = 0;
    std::string backend = "cpu";
};

struct InitLog {
    std::string text;
};

void capture_init_log(enum ggml_log_level, const char * text, void * user_data) {
    if (text == nullptr || user_data == nullptr) {
        return;
    }
    static_cast<InitLog *>(user_data)->text.append(text);
}

void throw_illegal_state(JNIEnv * env, const std::string & message) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type != nullptr) {
        env->ThrowNew(type, message.c_str());
    }
}

ParakeetEngine * require_engine(JNIEnv * env, jlong handle) {
    auto * engine = reinterpret_cast<ParakeetEngine *>(handle);
    if (engine == nullptr) {
        throw_illegal_state(env, "The Parakeet benchmark engine is closed.");
    }
    return engine;
}

jstring new_utf8_string(JNIEnv * env, const std::string & value) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(value.size()));
    if (bytes == nullptr) {
        return nullptr;
    }
    if (!value.empty()) {
        env->SetByteArrayRegion(
            bytes,
            0,
            static_cast<jsize>(value.size()),
            reinterpret_cast<const jbyte *>(value.data())
        );
    }
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID constructor =
        env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset_name = env->NewStringUTF("UTF-8");
    auto result = static_cast<jstring>(
        env->NewObject(string_class, constructor, bytes, charset_name)
    );
    env->DeleteLocalRef(charset_name);
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(bytes);
    return result;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_create(
    JNIEnv * env,
    jclass,
    jstring model_path,
    jint threads,
    jboolean use_gpu
) {
    if (threads < 1 || threads > 8) {
        throw_illegal_state(env, "Parakeet benchmark threads must be between 1 and 8.");
        return 0;
    }
    const char * characters = env->GetStringUTFChars(model_path, nullptr);
    const std::string path = characters == nullptr ? std::string() : characters;
    if (characters != nullptr) {
        env->ReleaseStringUTFChars(model_path, characters);
    }
    if (path.empty()) {
        throw_illegal_state(env, "Parakeet benchmark model path is empty.");
        return 0;
    }

    parakeet_context_params params = parakeet_context_default_params();
    params.use_gpu = use_gpu == JNI_TRUE;
    params.gpu_device = 0;
    InitLog init_log;
    parakeet_log_set(capture_init_log, &init_log);
    parakeet_context * context = parakeet_init_from_file_with_params(path.c_str(), params);
    parakeet_log_set(nullptr, nullptr);
    if (context == nullptr) {
        throw_illegal_state(env, "whisper.cpp could not load the staged Parakeet model.");
        return 0;
    }

    auto * engine = new ParakeetEngine();
    engine->context = context;
    engine->threads = threads;
    const std::string marker = "using ";
    const size_t at = init_log.text.find(marker);
    if (params.use_gpu && at != std::string::npos) {
        const size_t end = init_log.text.find(" backend", at);
        engine->backend = init_log.text.substr(at + marker.size(), end - at - marker.size());
    }
    if (params.use_gpu && engine->backend == "cpu") {
        parakeet_free(context);
        delete engine;
        throw_illegal_state(env, "Parakeet GPU was requested, but no GPU backend was activated.");
        return 0;
    }
    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_transcribe(
    JNIEnv * env,
    jclass,
    jlong handle,
    jfloatArray audio
) {
    ParakeetEngine * engine = require_engine(env, handle);
    if (engine == nullptr || audio == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->mutex);
    const jsize count = env->GetArrayLength(audio);
    std::vector<float> samples(static_cast<size_t>(count));
    env->GetFloatArrayRegion(audio, 0, count, samples.data());

    parakeet_full_params params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    params.n_threads = engine->threads;
    params.no_context = true;

    const auto started = std::chrono::steady_clock::now();
    const int status = parakeet_full(engine->context, params, samples.data(), count);
    engine->inference_millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started
    ).count();
    if (status != 0) {
        throw_illegal_state(env, "Parakeet inference failed with status " + std::to_string(status) + ".");
        return;
    }

    engine->token_triplets.clear();
    engine->pieces.clear();
    const int segments = parakeet_full_n_segments(engine->context);
    for (int segment = 0; segment < segments; ++segment) {
        const int tokens = parakeet_full_n_tokens(engine->context, segment);
        for (int token = 0; token < tokens; ++token) {
            const parakeet_token_data data =
                parakeet_full_get_token_data(engine->context, segment, token);
            engine->token_triplets.push_back(data.id);
            engine->token_triplets.push_back(data.frame_index);
            engine->token_triplets.push_back(data.duration_value);
            const char * piece = parakeet_full_get_token_text(engine->context, segment, token);
            engine->pieces.emplace_back(piece == nullptr ? "" : piece);
        }
    }
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_getTokenTriplets(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    ParakeetEngine * engine = require_engine(env, handle);
    if (engine == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(engine->mutex);
    jintArray result = env->NewIntArray(static_cast<jsize>(engine->token_triplets.size()));
    if (result != nullptr && !engine->token_triplets.empty()) {
        env->SetIntArrayRegion(
            result,
            0,
            static_cast<jsize>(engine->token_triplets.size()),
            engine->token_triplets.data()
        );
    }
    return result;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_getPieces(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    ParakeetEngine * engine = require_engine(env, handle);
    if (engine == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(engine->mutex);
    jclass string_class = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(
        static_cast<jsize>(engine->pieces.size()),
        string_class,
        nullptr
    );
    for (size_t index = 0; index < engine->pieces.size(); ++index) {
        jstring piece = new_utf8_string(env, engine->pieces[index]);
        env->SetObjectArrayElement(result, static_cast<jsize>(index), piece);
        env->DeleteLocalRef(piece);
    }
    env->DeleteLocalRef(string_class);
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_getInferenceMillis(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    ParakeetEngine * engine = require_engine(env, handle);
    return engine == nullptr ? 0 : engine->inference_millis;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_getBackend(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    ParakeetEngine * engine = require_engine(env, handle);
    return engine == nullptr ? nullptr : new_utf8_string(env, engine->backend);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_voicememo_transcription_ParakeetCppBenchmarkNative_destroy(
    JNIEnv *,
    jclass,
    jlong handle
) {
    auto * engine = reinterpret_cast<ParakeetEngine *>(handle);
    if (engine == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(engine->mutex);
        parakeet_free(engine->context);
        engine->context = nullptr;
    }
    delete engine;
}
