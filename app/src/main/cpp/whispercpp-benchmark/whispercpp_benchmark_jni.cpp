#include <jni.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "whisper.h"

namespace {

struct BenchmarkEngine {
    whisper_context * context = nullptr;
    int threads = 1;
    std::mutex mutex;
    std::atomic<bool> abort_requested{false};
    std::string text;
    std::string language = "und";
    std::vector<whisper_token> tokens;
    int64_t inference_millis = 0;
    int64_t encoder_millis = 0;
    std::string backend = "cpu";
    bool timestamp_tokens_enabled = false;
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

std::string vulkan_device_description() {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        const char * registry_name =
            registry == nullptr ? nullptr : ggml_backend_reg_name(registry);
        const auto type = ggml_backend_dev_type(device);
        if (registry_name != nullptr &&
            std::string(registry_name) == "Vulkan" &&
            (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
             type == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
            const char * name = ggml_backend_dev_name(device);
            const char * description = ggml_backend_dev_description(device);
            std::string result = name == nullptr ? "Vulkan" : name;
            if (description != nullptr && result != description) {
                result.append(" (");
                result.append(description);
                result.append(")");
            }
            return result;
        }
    }
    return {};
}

void throw_illegal_state(JNIEnv * env, const std::string & message) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type != nullptr) {
        env->ThrowNew(type, message.c_str());
    }
}

BenchmarkEngine * require_engine(JNIEnv * env, jlong handle) {
    auto * engine = reinterpret_cast<BenchmarkEngine *>(handle);
    if (engine == nullptr) {
        throw_illegal_state(env, "The whisper.cpp benchmark engine is closed.");
    }
    return engine;
}

bool abort_callback(void * data) {
    auto * engine = static_cast<BenchmarkEngine *>(data);
    return engine->abort_requested.load(std::memory_order_relaxed);
}

std::string copy_utf8(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char * characters = env->GetStringUTFChars(value, nullptr);
    if (characters == nullptr) {
        return {};
    }
    std::string result(characters);
    env->ReleaseStringUTFChars(value, characters);
    return result;
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
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_create(
    JNIEnv * env,
    jclass,
    jstring model_path,
    jint threads,
    jboolean use_vulkan,
    jboolean timestamp_tokens_enabled
) {
    if (threads < 1 || threads > 8) {
        throw_illegal_state(env, "whisper.cpp benchmark threads must be between 1 and 8.");
        return 0;
    }
    const std::string path = copy_utf8(env, model_path);
    if (path.empty()) {
        throw_illegal_state(env, "whisper.cpp benchmark model path is empty.");
        return 0;
    }

    const bool requested_vulkan = use_vulkan == JNI_TRUE;
    const std::string vulkan_device =
        requested_vulkan ? vulkan_device_description() : std::string();
    if (requested_vulkan && vulkan_device.empty()) {
        throw_illegal_state(
            env,
            "whisper.cpp Vulkan was requested, but no Vulkan GPU was registered."
        );
        return 0;
    }

    whisper_context_params context_params = whisper_context_default_params();
    context_params.use_gpu = requested_vulkan;
    context_params.gpu_device = 0;
    context_params.flash_attn = false;
    InitLog init_log;
    whisper_log_set(capture_init_log, &init_log);
    whisper_context * context =
        whisper_init_from_file_with_params(path.c_str(), context_params);
    whisper_log_set(nullptr, nullptr);
    if (context == nullptr) {
        throw_illegal_state(env, "whisper.cpp could not load the staged GGML model.");
        return 0;
    }
    if (requested_vulkan &&
        init_log.text.find("using Vulkan") == std::string::npos) {
        whisper_free(context);
        throw_illegal_state(
            env,
            "whisper.cpp did not activate its Vulkan backend; CPU fallback is rejected."
        );
        return 0;
    }

    auto * engine = new BenchmarkEngine();
    engine->context = context;
    engine->threads = threads;
    engine->backend =
        requested_vulkan ? "vulkan:" + vulkan_device : "cpu";
    engine->timestamp_tokens_enabled = timestamp_tokens_enabled == JNI_TRUE;
    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_transcribe(
    JNIEnv * env,
    jclass,
    jlong handle,
    jfloatArray audio,
    jstring language
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return;
    }
    if (audio == nullptr) {
        throw_illegal_state(env, "whisper.cpp benchmark audio is missing.");
        return;
    }

    const jsize sample_count = env->GetArrayLength(audio);
    if (sample_count <= 0 || sample_count > WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE) {
        throw_illegal_state(
            env,
            "whisper.cpp benchmark audio must contain between 1 sample and 30 seconds."
        );
        return;
    }
    std::vector<float> samples(static_cast<size_t>(sample_count));
    env->GetFloatArrayRegion(audio, 0, sample_count, samples.data());
    if (env->ExceptionCheck()) {
        return;
    }
    const std::string requested_language = copy_utf8(env, language);

    std::lock_guard<std::mutex> guard(engine->mutex);
    engine->abort_requested.store(false, std::memory_order_relaxed);
    engine->text.clear();
    engine->tokens.clear();
    engine->language = "und";
    engine->inference_millis = 0;
    whisper_reset_timings(engine->context);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = engine->threads;
    params.translate = false;
    params.no_context = true;
    params.no_timestamps = !engine->timestamp_tokens_enabled;
    params.single_segment = false;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.token_timestamps = false;
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;
    params.greedy.best_of = 1;
    params.language =
        requested_language.empty() || requested_language == "auto"
            ? nullptr
            : requested_language.c_str();
    params.detect_language = false;
    params.abort_callback = abort_callback;
    params.abort_callback_user_data = engine;

    const auto started = std::chrono::steady_clock::now();
    const int status = whisper_full(
        engine->context,
        params,
        samples.data(),
        static_cast<int>(samples.size())
    );
    const auto finished = std::chrono::steady_clock::now();
    engine->inference_millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();

    if (status != 0) {
        throw_illegal_state(
            env,
            engine->abort_requested.load(std::memory_order_relaxed)
                ? "whisper.cpp benchmark transcription was cancelled."
                : "whisper.cpp benchmark transcription failed."
        );
        return;
    }

    const int language_id = whisper_full_lang_id(engine->context);
    const char * language_code = whisper_lang_str(language_id);
    if (language_code != nullptr) {
        engine->language = language_code;
    }

    const whisper_token end_of_text = whisper_token_eot(engine->context);
    const whisper_token timestamp_begin = whisper_token_beg(engine->context);
    const int segment_count = whisper_full_n_segments(engine->context);
    for (int segment = 0; segment < segment_count; ++segment) {
        const char * segment_text =
            whisper_full_get_segment_text(engine->context, segment);
        if (segment_text != nullptr) {
            engine->text.append(segment_text);
        }
        const int token_count = whisper_full_n_tokens(engine->context, segment);
        for (int token_index = 0; token_index < token_count; ++token_index) {
            const whisper_token token =
                whisper_full_get_token_id(engine->context, segment, token_index);
            if (
                token >= 0 &&
                (
                    token < end_of_text ||
                    (engine->timestamp_tokens_enabled && token >= timestamp_begin)
                )
            ) {
                engine->tokens.push_back(token);
            }
        }
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_encodeMel(
    JNIEnv * env,
    jclass,
    jlong handle,
    jfloatArray mel,
    jint frame_count,
    jint mel_bin_count
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    if (mel == nullptr || frame_count <= 0 || mel_bin_count <= 0) {
        throw_illegal_state(env, "whisper.cpp encoder diagnostic mel input is invalid.");
        return nullptr;
    }
    const jsize value_count = env->GetArrayLength(mel);
    if (value_count != frame_count * mel_bin_count) {
        throw_illegal_state(env, "whisper.cpp encoder diagnostic mel shape is invalid.");
        return nullptr;
    }
    std::vector<float> values(static_cast<size_t>(value_count));
    env->GetFloatArrayRegion(mel, 0, value_count, values.data());
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> guard(engine->mutex);
    whisper_enable_encoder_block_diagnostics(engine->context, true);
    if (whisper_set_mel(engine->context, values.data(), frame_count, mel_bin_count) != 0) {
        throw_illegal_state(env, "whisper.cpp rejected the shared mel tensor.");
        return nullptr;
    }
    const auto started = std::chrono::steady_clock::now();
    if (whisper_encode(engine->context, 0, engine->threads) != 0) {
        throw_illegal_state(env, "whisper.cpp encoder diagnostic failed.");
        return nullptr;
    }
    const auto finished = std::chrono::steady_clock::now();
    engine->encoder_millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();

    const int element_count = whisper_encoder_state_n_elements(engine->context);
    const int context_count = whisper_encoder_state_n_ctx(engine->context);
    const int state_count = whisper_encoder_state_n_state(engine->context);
    if (element_count <= 0 || context_count * state_count != element_count) {
        throw_illegal_state(env, "whisper.cpp returned an invalid encoder-state shape.");
        return nullptr;
    }
    std::vector<float> encoder_state(static_cast<size_t>(element_count));
    if (whisper_copy_encoder_state(
            engine->context,
            encoder_state.data(),
            element_count) != 0) {
        throw_illegal_state(env, "whisper.cpp could not copy its final FP32 encoder state.");
        return nullptr;
    }
    jfloatArray result = env->NewFloatArray(element_count);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, element_count, encoder_state.data());
    }
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getEncoderBlockCount(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    return whisper_encoder_block_diagnostic_count(engine->context);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getEncoderBlockState(
    JNIEnv * env,
    jclass,
    jlong handle,
    jint index
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    const int element_count = whisper_encoder_state_n_elements(engine->context);
    if (element_count <= 0) {
        throw_illegal_state(env, "whisper.cpp encoder block shape is unavailable.");
        return nullptr;
    }
    std::vector<float> state(static_cast<size_t>(element_count));
    if (whisper_copy_encoder_block_diagnostic(
            engine->context,
            index,
            state.data(),
            element_count) != 0) {
        throw_illegal_state(env, "whisper.cpp encoder block copy failed.");
        return nullptr;
    }
    jfloatArray result = env->NewFloatArray(element_count);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, element_count, state.data());
    }
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getEncoderMillis(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    return engine->encoder_millis;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getText(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    return new_utf8_string(env, engine->text);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getLanguage(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    return env->NewStringUTF(engine->language.c_str());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getTokens(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    jintArray result = env->NewIntArray(static_cast<jsize>(engine->tokens.size()));
    if (result != nullptr && !engine->tokens.empty()) {
        env->SetIntArrayRegion(
            result,
            0,
            static_cast<jsize>(engine->tokens.size()),
            reinterpret_cast<const jint *>(engine->tokens.data())
        );
    }
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_getInferenceMillis(
    JNIEnv * env,
    jclass,
    jlong handle
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return 0;
    }
    std::lock_guard<std::mutex> guard(engine->mutex);
    return engine->inference_millis;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_decode(
    JNIEnv * env,
    jclass,
    jlong handle,
    jintArray tokens
) {
    BenchmarkEngine * engine = require_engine(env, handle);
    if (engine == nullptr || env->ExceptionCheck()) {
        return nullptr;
    }
    if (tokens == nullptr) {
        return env->NewStringUTF("");
    }
    const jsize count = env->GetArrayLength(tokens);
    std::vector<jint> token_ids(static_cast<size_t>(count));
    env->GetIntArrayRegion(tokens, 0, count, token_ids.data());
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> guard(engine->mutex);
    std::string text;
    const whisper_token end_of_text = whisper_token_eot(engine->context);
    for (const jint token : token_ids) {
        if (token < 0 || token >= end_of_text) {
            continue;
        }
        const char * piece = whisper_token_to_str(engine->context, token);
        if (piece != nullptr) {
            text.append(piece);
        }
    }
    return new_utf8_string(env, text);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_runtimeInfo(
    JNIEnv * env,
    jclass
) {
    std::string info = "whisper.cpp ";
    info.append(whisper_version());
    info.append(" revision ");
    info.append(VOICEMEMO_WHISPER_CPP_REVISION);
    info.append(" | ");
    info.append(whisper_print_system_info());
    const std::string vulkan_device = vulkan_device_description();
    info.append(" | Vulkan=");
    info.append(vulkan_device.empty() ? "unavailable" : vulkan_device);
    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_voicememo_transcription_WhisperCppBenchmarkNative_destroy(
    JNIEnv *,
    jclass,
    jlong handle
) {
    auto * engine = reinterpret_cast<BenchmarkEngine *>(handle);
    if (engine == nullptr) {
        return;
    }
    engine->abort_requested.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> guard(engine->mutex);
        whisper_free(engine->context);
        engine->context = nullptr;
    }
    delete engine;
}
