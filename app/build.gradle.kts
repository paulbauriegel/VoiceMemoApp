import java.security.MessageDigest
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.kapt")
}

val whisperWorkerCount = providers.gradleProperty("whisperWorkerCount")
    .orElse("2")
    .map(String::toInt)
val whisperThreadsPerWorker = providers.gradleProperty("whisperThreadsPerWorker")
    .orElse("3")
    .map(String::toInt)
val whisperBackend = providers.gradleProperty("whisperBackend")
    .orElse("cpu")
    .map { it.trim().lowercase() }
val includeDecoderSplitDiagnostics = providers
    .gradleProperty("includeDecoderSplitDiagnostics")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeDecoderSplitDiagnostics must be true or false.")
    }
val includeRewrittenDecoderPrefixDiagnostic = providers
    .gradleProperty("includeRewrittenDecoderPrefixDiagnostic")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeRewrittenDecoderPrefixDiagnostic must be true or false.")
    }
val includeInt8FcRegressionDiagnostic = providers
    .gradleProperty("includeInt8FcRegressionDiagnostic")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeInt8FcRegressionDiagnostic must be true or false.")
    }
val includeInt8FcArithmeticDiagnostic = providers
    .gradleProperty("includeInt8FcArithmeticDiagnostic")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeInt8FcArithmeticDiagnostic must be true or false.")
    }
val includeFullFloatDecoderDiagnostic = providers
    .gradleProperty("includeFullFloatDecoderDiagnostic")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeFullFloatDecoderDiagnostic must be true or false.")
    }
val includeWhisperCppBenchmark = providers
    .gradleProperty("includeWhisperCppBenchmark")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeWhisperCppBenchmark must be true or false.")
    }
val includeEncoderRuntimeParityDiagnostic = providers
    .gradleProperty("includeEncoderRuntimeParityDiagnostic")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("includeEncoderRuntimeParityDiagnostic must be true or false.")
    }
val useFullFloatDecoderProduction = providers
    .gradleProperty("useFullFloatDecoderProduction")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("useFullFloatDecoderProduction must be true or false.")
    }
val useSplitWhisperProduction = providers
    .gradleProperty("useSplitWhisperProduction")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("useSplitWhisperProduction must be true or false.")
    }
val rewrittenDecoderPrefixStage = providers
    .gradleProperty("rewrittenDecoderPrefixStage")
    .orElse("after-position-add")
    .map { value ->
        require(
            value in setOf(
                "after-position-add",
                "after-self-layer-norm",
                "after-self-key-projection",
                "after-self-query-projection",
                "after-self-query-reshape",
                "after-self-key-transpose",
                "after-self-score-matmul",
                "after-self-mask-add",
                "after-self-softmax",
                "after-self-value-projection",
                "after-self-value-reshape",
                "after-self-value-matmul",
                "after-self-context-reshape",
                "after-self-output-projection",
                "after-self-attention",
                "after-cross-attention",
                "after-1-block",
                "after-2-blocks",
                "after-3-blocks",
                "after-4-blocks",
                "after-final-layer-norm",
            ),
        ) {
            "rewrittenDecoderPrefixStage is not a supported decoder boundary."
        }
        value
    }
val rewrittenDecoderPrefixSha256 = providers
    .gradleProperty("rewrittenDecoderPrefixSha256")
    .orElse("")
    .map(String::lowercase)
val rewrittenDecoderPrefixBytes = providers
    .gradleProperty("rewrittenDecoderPrefixBytes")
    .orElse("1094571944")
    .map { value ->
        value.toLongOrNull()?.also {
            require(it in 1_000_000_000L..1_200_000_000L) {
                "rewrittenDecoderPrefixBytes is outside the fail-closed diagnostic range."
            }
        } ?: error("rewrittenDecoderPrefixBytes must be an integer.")
    }
val useStrictMathGpuProbe = providers
    .gradleProperty("useStrictMathGpuProbe")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("useStrictMathGpuProbe must be true or false.")
    }
val useFp16Fp32AccumGpuProbe = providers
    .gradleProperty("useFp16Fp32AccumGpuProbe")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("useFp16Fp32AccumGpuProbe must be true or false.")
    }
val installReleaseLocally = providers
    .gradleProperty("installReleaseLocally")
    .orElse("false")
    .map { value ->
        value.toBooleanStrictOrNull()
            ?: error("installReleaseLocally must be true or false.")
    }
val decoderSplitDiagnosticStage = providers
    .gradleProperty("decoderSplitDiagnosticStage")
    .orElse("post-layernorm")
    .map { value ->
        require(
            value in setOf(
                "post-layernorm",
                "pre-layernorm",
                "after-position-add",
                "after-cross-attention",
                "after-1-block",
                "after-2-blocks",
                "after-3-blocks",
            ),
        ) {
            "decoderSplitDiagnosticStage must be post-layernorm, pre-layernorm, " +
                "after-position-add, after-cross-attention, after-1-block, " +
                "after-2-blocks, or after-3-blocks."
        }
        value
    }
val decoderSplitDiagnosticDirectory =
    rootProject.layout.projectDirectory.dir("build/generated/whisper-decoder-split")
val decoderSplitDiagnosticAssetRoot =
    layout.buildDirectory.dir("generated/decoderSplitAndroidTestAssets")
val rewrittenDecoderPrefixFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/decoder-prefix.tflite",
)
val rewrittenDecoderSoftmaxProbeFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/softmax-probe.tflite",
)
val rewrittenDecoderSoftmaxProbeSha256 =
    "0dda1b63e41cb5aff06f1a0141ccefc3b851402119ca04315f02e829af1258b1"
val rewrittenDecoderMaskAddProbeFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/mask-add-probe.tflite",
)
val rewrittenDecoderMaskAddProbeSha256 =
    "b55f04c5c4667bdb9333b79b327cfde35493fc217ffe186f8087f66965fc213d"
val rewrittenDecoderScoreProbeFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/score-probe.tflite",
)
val rewrittenDecoderScoreProbeSha256 =
    "358ecf5bef1d39b739dfc54125a5a6d5273898adc7784625757485717a280dfd"
val rewrittenDecoderValueProbeFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/value-probe.tflite",
)
val rewrittenDecoderValueProbeSha256 =
    "6abd8afd394af3005e7304432e85e78d5058193fa9bd5ae7d00d7e4315fea861"
val rewrittenDecoderKProjectionProbeFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-decoder-prefix/k-projection-probe.tflite",
)
val rewrittenDecoderKProjectionProbeSha256 =
    "7a30331ccb89e332edb176587d149f8d852e5b224864247defadae37db940b1d"
val int8FcRegressionFixtureFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-int8-fc/" +
        "whisper-first-block-k-projection-step-7-input.f32",
)
val int8FcRegressionFixtureSha256 =
    "105b86c9c20463e3920d871f0f2a653a9da88ce8739572f489e077754f6c29cb"
val rewrittenDecoderPrefixAssetRoot =
    layout.buildDirectory.dir("generated/rewrittenDecoderPrefixAndroidTestAssets")
val int8FcRegressionAssetRoot =
    layout.buildDirectory.dir("generated/int8FcRegressionAndroidTestAssets")
val int8FcArithmeticDirectory =
    rootProject.layout.projectDirectory.dir("build/generated/whisper-int8-fc-matrix")
val int8FcArithmeticManifestSha256 =
    "89d2b1777e4abd25dc22ca405be0014423f94cdf4460b1dd018747ab2b92f94a"
val int8FcArithmeticAssetRoot =
    layout.buildDirectory.dir("generated/int8FcArithmeticAndroidTestAssets")
val fullFloatDecoderDiagnosticFile = rootProject.layout.projectDirectory.file(
    "build/generated/whisper-full-float-decoder/" +
        "whisper_large_v3_turbo_full_float_decoder.tflite",
)
val fullFloatDecoderDiagnosticBytes = 1_779_622_096L
val fullFloatDecoderDiagnosticSha256 =
    "a2613d64e6cb73ff40db7c548215fd086f096aa088c6cc113c833ccc87501e89"
val fullFloatDecoderDiagnosticAssetRoot =
    layout.buildDirectory.dir("generated/fullFloatDecoderAndroidTestAssets")
val fullFloatDecoderProductionAssetRoot =
    layout.buildDirectory.dir("generated/fullFloatDecoderProductionAssets")
val splitWhisperProductionAssetRoot =
    layout.buildDirectory.dir("generated/splitWhisperProductionAssets")
val splitWhisperEncoderFile = rootProject.layout.projectDirectory.file(
    "model-assets/huggingface/whisper_large_v3_turbo_split_encoder.tflite",
)
val splitWhisperDecoderFile = rootProject.layout.projectDirectory.file(
    "model-assets/huggingface/whisper_large_v3_turbo_split_decoder.tflite",
)
// FLOAT16-storage corrected encoder (LayerNorm-safe graph, FP16 weights on
// disk). Token-identical to the FLOAT32-storage encoder on the five-sample
// gate because the GPU delegate rounds weights to FP16 at compile time either
// way; halves the bundled encoder from 2.55 GB to 1.27 GB.
val splitWhisperEncoderBytes = 1_274_327_104L
val splitWhisperEncoderSha256 =
    "faf331b520fbf67896e0080b8f7803037d32b57987ec914f93e185c3692c3fc1"
val splitWhisperDecoderBytes = 951_674_764L
val splitWhisperDecoderSha256 =
    "d661702d7753165cfaf01bf5b060f46065df500959e42e5254fd2598a54dedb7"
// Stateful KV-cache decoder assets (FLOAT16-stored step + cross-cache, exact
// FLOAT32 vocabulary projection). Same artifact hashes as the five-sample
// benchmarked research models.
val kvCrossCacheFile = rootProject.layout.projectDirectory.file(
    "model-assets/huggingface/whisper_large_v3_turbo_kv_cross_cache.tflite",
)
val kvStepFile = rootProject.layout.projectDirectory.file(
    "model-assets/huggingface/whisper_large_v3_turbo_kv_step.tflite",
)
val kvProjectionFile = rootProject.layout.projectDirectory.file(
    "model-assets/huggingface/whisper_large_v3_turbo_kv_projection_fp32.tflite",
)
val kvCrossCacheBytes = 26_231_580L
val kvCrossCacheSha256 =
    "d3402e3b1aa9b2cece3b28d2919a1e3cca3ac9655c0f0a983a58d14b937b7b21"
val kvStepBytes = 317_655_916L
val kvStepSha256 =
    "13ed9528e5df1ad62582f25fe8677bc63f2c26812017e84b80e015d5aa394b7a"
val kvProjectionBytes = 265_554_672L
val kvProjectionSha256 =
    "ba1eef8fed0d2edbd33fffafc2c7dfd0ac408ca2d8ed2262fec1f1a430dffd7b"
val kvCrossCacheAssetPath = "whisper/whisper_large_v3_turbo_kv_cross_cache.tflite"
val kvStepAssetPath = "whisper/whisper_large_v3_turbo_kv_step.tflite"
val kvProjectionAssetPath = "whisper/whisper_large_v3_turbo_kv_projection_fp32.tflite"
// sha256("split-whisper:encoder=<enc>:decoder=<dec>:kvstep=<step>:" +
//        "kvcross=<cross>:kvproj=<proj>") — a configuration-identity marker
// for transcript caching; changes whenever any bundled model changes.
val splitWhisperConfigurationSha256 =
    "a3f8f389c8ca0c221c68ccbf0e0a1289e8d2497eae856403c3c3d92a6128f7e8"
val parakeetProductionAssetRoot =
    layout.buildDirectory.dir("generated/parakeetProductionAssets")
val parakeetSourceAssetRoot =
    layout.projectDirectory.dir("src/main/generatedAssets/parakeet")
val graniteMetadataAssetRoot =
    layout.projectDirectory.dir("src/main/generatedAssets/granite")
val graniteMetadataProductionAssetRoot =
    layout.buildDirectory.dir("generated/graniteMetadataProductionAssets")
val whisperCppRevision = "f049fff95a089aa9969deb009cdd4892b3e74916"
val whisperCppSourceRoot =
    rootProject.layout.projectDirectory.dir("build/third_party/whisper.cpp-v1.9.1")
val whisperCppVulkanHeadersRoot =
    rootProject.layout.projectDirectory.dir("build/third_party/Vulkan-Headers-1.4.350.1")
val fullFloatDecoderProductionAssetPath =
    "whisper/whisper_large_v3_turbo_full_float_decoder.tflite"
val splitWhisperEncoderAssetPath =
    "whisper/whisper_large_v3_turbo_split_encoder.tflite"
val splitWhisperDecoderAssetPath =
    "whisper/whisper_large_v3_turbo_split_decoder.tflite"
val sileroVadAssetRoot =
    layout.projectDirectory.dir("src/main/generatedVadAssets")
val strictMathGpuProbeAar = rootProject.layout.projectDirectory.file(
    "build/generated/litert-strict-math-probe/litert-2.1.6-strict-math-probe.aar",
)
val strictMathGpuProbeExpectedBytes = 33_633_920L
val strictMathGpuProbeExpectedSha256 =
    "322ba49f63712569ece6d7ead0ec1c6075d55766728e4de52be1f5a327b8ffc3"
val strictMathGpuProbeAcceleratorEntry =
    "jni/arm64-v8a/libLiteRtClGlAccelerator.so"
val strictMathGpuProbePatchedAcceleratorSha256 =
    "9dbc2ff9a9c2b0297fecb3fb5ac48df0442c0707224627a2e76ac3270b0d89b6"
val strictMathGpuProbeFlag = "-cl-fast-relaxed-math".toByteArray(Charsets.US_ASCII)
val strictMathGpuProbeFlagOffset = 0x510c7
val fp16Fp32AccumGpuProbeAar = rootProject.layout.projectDirectory.file(
    "build/generated/litert-fp16-fp32-accum-probe/" +
        "litert-2.1.6-fp16-fp32-accum-probe.aar",
)
val fp16Fp32AccumGpuProbeExpectedBytes = 33_633_920L
val fp16Fp32AccumGpuProbeExpectedSha256 =
    "2582c44a2a36043d66fc812005fcfae7b85bc71885366245c1a0b2ddd75be29d"
val fp16Fp32AccumGpuProbeAcceleratorEntry =
    "jni/arm64-v8a/libLiteRtClGlAccelerator.so"
val fp16Fp32AccumGpuProbeAcceleratorSha256 =
    "360192dd650ee085c3ae3a393232a36ea586fe1fb588a2f97888943a3689dc34"
val fp16Fp32AccumGpuProbeOffset = 0xeb518
val fp16Fp32AccumGpuProbeInstruction = byteArrayOf(0x2a, 0x00, 0x80.toByte(), 0x52)

require(whisperWorkerCount.get() in 1..2) {
    "whisperWorkerCount must be 1 or 2."
}
require(whisperThreadsPerWorker.get() in 1..8) {
    "whisperThreadsPerWorker must be between 1 and 8."
}
require(whisperBackend.get() in setOf("cpu", "gpu")) {
    "whisperBackend must be cpu or gpu."
}
require(
    !useFullFloatDecoderProduction.get() ||
        (
            whisperBackend.get() == "gpu" &&
                whisperWorkerCount.get() == 1
            )
) {
    """
    useFullFloatDecoderProduction requires whisperBackend=gpu and
    whisperWorkerCount=1 because one rewritten GPU model peaks above 4 GB PSS.
    """.trimIndent()
}
require(!(useFullFloatDecoderProduction.get() && useSplitWhisperProduction.get())) {
    "Combined and split Whisper production models are mutually exclusive."
}
require(
    !useSplitWhisperProduction.get() ||
        (whisperBackend.get() == "gpu" && whisperWorkerCount.get() == 1),
) {
    "useSplitWhisperProduction requires whisperBackend=gpu and whisperWorkerCount=1."
}
require(
    !useStrictMathGpuProbe.get() ||
        includeDecoderSplitDiagnostics.get() ||
        includeEncoderRuntimeParityDiagnostic.get(),
) {
    """
    useStrictMathGpuProbe is a diagnostic-only dependency substitution.
    Re-run with -PuseStrictMathGpuProbe=true and either
    -PincludeDecoderSplitDiagnostics=true or
    -PincludeEncoderRuntimeParityDiagnostic=true.
    """.trimIndent()
}
require(!includeEncoderRuntimeParityDiagnostic.get() || includeWhisperCppBenchmark.get()) {
    "includeEncoderRuntimeParityDiagnostic requires includeWhisperCppBenchmark."
}
require(!(useStrictMathGpuProbe.get() && useFp16Fp32AccumGpuProbe.get())) {
    "Only one diagnostic LiteRT AAR substitution may be enabled."
}

android {
    namespace = "com.example.voicememo"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.voicememo"
        // Silero VAD runs through the official ONNX Runtime Android package,
        // whose supported Android floor is API 24.
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        buildConfigField("int", "WHISPER_WORKER_COUNT", whisperWorkerCount.get().toString())
        buildConfigField(
            "int",
            "WHISPER_THREADS_PER_WORKER",
            whisperThreadsPerWorker.get().toString(),
        )
        buildConfigField(
            "String",
            "WHISPER_BACKEND_REQUESTED",
            "\"${whisperBackend.get()}\"",
        )
        buildConfigField(
            "String",
            "WHISPER_MODEL_ASSET_PATH",
            if (useSplitWhisperProduction.get()) {
                "\"$splitWhisperDecoderAssetPath\""
            } else if (useFullFloatDecoderProduction.get()) {
                "\"$fullFloatDecoderProductionAssetPath\""
            } else {
                "\"whisper/whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite\""
            },
        )
        buildConfigField(
            "long",
            "WHISPER_MODEL_SIZE_BYTES",
            if (useSplitWhisperProduction.get()) {
                "${splitWhisperDecoderBytes}L"
            } else if (useFullFloatDecoderProduction.get()) {
                "${fullFloatDecoderDiagnosticBytes}L"
            } else {
                "1094571944L"
            },
        )
        buildConfigField(
            "String",
            "WHISPER_MODEL_SHA256",
            if (useSplitWhisperProduction.get()) {
                "\"$splitWhisperConfigurationSha256\""
            } else if (useFullFloatDecoderProduction.get()) {
                "\"$fullFloatDecoderDiagnosticSha256\""
            } else {
                "\"8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418\""
            },
        )
        buildConfigField(
            "boolean",
            "WHISPER_SPLIT_MODEL_PRODUCTION",
            useSplitWhisperProduction.get().toString(),
        )
        buildConfigField(
            "boolean",
            "WHISPER_KV_DECODER_PRODUCTION",
            useSplitWhisperProduction.get().toString(),
        )
        buildConfigField(
            "String",
            "WHISPER_KV_CROSS_CACHE_ASSET_PATH",
            "\"$kvCrossCacheAssetPath\"",
        )
        buildConfigField(
            "String",
            "WHISPER_KV_STEP_ASSET_PATH",
            "\"$kvStepAssetPath\"",
        )
        buildConfigField(
            "String",
            "WHISPER_KV_PROJECTION_ASSET_PATH",
            "\"$kvProjectionAssetPath\"",
        )
        buildConfigField(
            "String",
            "WHISPER_ENCODER_MODEL_ASSET_PATH",
            "\"$splitWhisperEncoderAssetPath\"",
        )
        buildConfigField(
            "long",
            "WHISPER_ENCODER_MODEL_SIZE_BYTES",
            "${splitWhisperEncoderBytes}L",
        )
        buildConfigField(
            "String",
            "WHISPER_ENCODER_MODEL_SHA256",
            "\"$splitWhisperEncoderSha256\"",
        )
        buildConfigField(
            "String",
            "WHISPER_DECODER_MODEL_ASSET_PATH",
            "\"$splitWhisperDecoderAssetPath\"",
        )
        buildConfigField(
            "long",
            "WHISPER_DECODER_MODEL_SIZE_BYTES",
            "${splitWhisperDecoderBytes}L",
        )
        buildConfigField(
            "String",
            "WHISPER_DECODER_MODEL_SHA256",
            "\"$splitWhisperDecoderSha256\"",
        )
        buildConfigField(
            "boolean",
            "WHISPER_CPP_BENCHMARK_INCLUDED",
            includeWhisperCppBenchmark.get().toString(),
        )
        if (includeWhisperCppBenchmark.get()) {
            externalNativeBuild {
                cmake {
                    arguments += listOf(
                        "-DWHISPER_CPP_SOURCE_DIR=${whisperCppSourceRoot.asFile.absolutePath}",
                        "-DWHISPER_CPP_REVISION=$whisperCppRevision",
                        "-DVULKAN_HEADERS_DIR=${whisperCppVulkanHeadersRoot.asFile.absolutePath}",
                    )
                    cppFlags += listOf("-std=c++17", "-O3")
                }
            }
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables.useSupportLibrary = true

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    buildTypes {
        getByName("release") {
            // Normal release builds remain unsigned. The shared Android Studio
            // "Install Release" configuration opts into the debug key so AGP
            // can update the locally installed app without deleting its data.
            if (installReleaseLocally.get()) {
                signingConfig = signingConfigs.getByName("debug")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main").assets.directories.add(sileroVadAssetRoot.asFile.absolutePath)
        if (useFullFloatDecoderProduction.get() || useSplitWhisperProduction.get()) {
            if (useFullFloatDecoderProduction.get()) {
                getByName("main").assets.directories.add(
                    fullFloatDecoderProductionAssetRoot.get().asFile.absolutePath,
                )
            } else {
                getByName("main").assets.directories.add(
                    splitWhisperProductionAssetRoot.get().asFile.absolutePath,
                )
            }
            getByName("main").assets.directories.add(
                parakeetProductionAssetRoot.get().asFile.absolutePath,
            )
            // Specialized Whisper production replaces the general generated-assets
            // root. Keep the other production models explicit in this build mode.
            getByName("main").assets.directories.add(
                graniteMetadataProductionAssetRoot.get().asFile.absolutePath,
            )
        } else {
            getByName("main").assets.directories.add("src/main/generatedAssets")
        }
        if (includeDecoderSplitDiagnostics.get()) {
            // Large generated models are packaged only into the opt-in
            // instrumentation APK. They never replace production app assets.
            getByName("androidTest").assets.directories.add(
                decoderSplitDiagnosticAssetRoot.get().asFile.absolutePath,
            )
        }
        if (includeRewrittenDecoderPrefixDiagnostic.get()) {
            // This fail-closed prefix model is diagnostic-only and is never
            // included in the production APK.
            getByName("androidTest").assets.directories.add(
                rewrittenDecoderPrefixAssetRoot.get().asFile.absolutePath,
            )
        }
        if (includeInt8FcRegressionDiagnostic.get()) {
            // The exact 1.65 MB INT8 K projection is independently packageable
            // so its GPU regression does not require the 1.1 GB prefix model.
            getByName("androidTest").assets.directories.add(
                int8FcRegressionAssetRoot.get().asFile.absolutePath,
            )
        }
        if (includeInt8FcArithmeticDiagnostic.get()) {
            // Synthetic and requantized K-projection variants isolate scale,
            // zero-point, and accumulation behavior in LiteRT's GPU INT8 FC path.
            getByName("androidTest").assets.directories.add(
                int8FcArithmeticAssetRoot.get().asFile.absolutePath,
            )
        }
        if (includeFullFloatDecoderDiagnostic.get()) {
            // Complete-decoder FLOAT32 weights and mask-rank barriers remain
            // isolated to the opt-in instrumentation APK.
            getByName("androidTest").assets.directories.add(
                fullFloatDecoderDiagnosticAssetRoot.get().asFile.absolutePath,
            )
        }
    }

    androidResources {
        // LiteRT maps the model directly from the APK file descriptor. Deflating
        // this entry would require another ~1.09 GB copy and prevent mmap.
        noCompress += "tflite"
        noCompress += "onnx"
        noCompress += "litertlm"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
        resources {
            excludes += setOf(
                "/META-INF/{AL2.0,LGPL2.1}",
                "/META-INF/DEPENDENCIES",
            )
        }
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }

    lint {
        abortOnError = true
        checkReleaseBuilds = true
    }

    if (includeWhisperCppBenchmark.get()) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/whispercpp-benchmark/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        freeCompilerArgs.addAll(
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
            "-opt-in=androidx.compose.material3.ExperimentalMaterial3Api",
        )
    }
}

kapt {
    correctErrorTypes = true
    arguments {
        arg("room.schemaLocation", "$projectDir/schemas")
        arg("room.incremental", "true")
        arg("room.expandProjection", "true")
    }
}

val whisperAssetDirectory = layout.projectDirectory.dir("src/main/generatedAssets/whisper")
val requiredWhisperAssets = mapOf(
    "whisper_large_v3_turbo_30s_i8_explicit_key_transpose.tflite" to (
        1_094_571_944L to "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418"
        ),
    "tokenizer.json" to (
        2_710_337L to "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"
        ),
    "generation_config.json" to (
        3_772L to "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"
        ),
    "preprocessor_config.json" to (
        340L to "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
        ),
)

val verifyWhisperAssets by tasks.registering {
    group = "verification"
    description = "Checks that the pinned offline Whisper assets have been fetched and verified."
    inputs.files(requiredWhisperAssets.keys.map { whisperAssetDirectory.file(it) })

    doLast {
        val missing = requiredWhisperAssets.keys
            .map { whisperAssetDirectory.file(it).asFile }
            .filterNot { it.isFile }
        check(missing.isEmpty()) {
            """
            Offline Whisper assets are missing:
            ${missing.joinToString(separator = "\n") { " - ${it.relativeTo(projectDir)}" }}

            Run scripts/fetch-whisper-assets.sh, then build again. The script downloads
            revision 4e3af6a1597a54b6e47d2899cf4f6c39932b0b7e and verifies every asset.
            """.trimIndent()
        }

        requiredWhisperAssets.forEach { (fileName, expected) ->
            val (expectedBytes, expectedSha256) = expected
            val assetFile = whisperAssetDirectory.file(fileName).asFile
            check(assetFile.length() == expectedBytes) {
                "$fileName has ${assetFile.length()} bytes; expected $expectedBytes. Re-run scripts/fetch-whisper-assets.sh."
            }
            val digest = MessageDigest.getInstance("SHA-256")
            assetFile.inputStream().buffered().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
            check(actualSha256 == expectedSha256) {
                "$fileName SHA-256 is $actualSha256; expected $expectedSha256. Delete generatedAssets and fetch again."
            }
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn(verifyWhisperAssets)
}

val verifyWhisperCppBenchmarkSource by tasks.registering {
    group = "verification"
    description = "Checks the pinned, opt-in whisper.cpp benchmark source checkout."
    onlyIf { includeWhisperCppBenchmark.get() }
    inputs.file(whisperCppSourceRoot.file(".voicememo-whispercpp-revision"))
    inputs.file(whisperCppSourceRoot.file("CMakeLists.txt"))
    inputs.file(whisperCppSourceRoot.file("include/whisper.h"))
    inputs.file(whisperCppVulkanHeadersRoot.file("include/vulkan/vulkan.hpp"))

    doLast {
        val sourceDirectory = whisperCppSourceRoot.asFile
        val revisionFile = File(sourceDirectory, ".voicememo-whispercpp-revision")
        check(File(sourceDirectory, "CMakeLists.txt").isFile) {
            """
            Pinned whisper.cpp benchmark sources are missing:
             - ${sourceDirectory.relativeTo(rootProject.projectDir)}

            Run scripts/fetch-whisper-cpp-benchmark.sh source, then rebuild with
            -PincludeWhisperCppBenchmark=true.
            """.trimIndent()
        }
        check(revisionFile.isFile && revisionFile.readText().trim() == whisperCppRevision) {
            "whisper.cpp source revision is missing or not pinned to $whisperCppRevision."
        }
        check(
            whisperCppVulkanHeadersRoot.file("include/vulkan/vulkan.hpp").asFile.isFile,
        ) {
            "Pinned Vulkan C++ headers are missing; run scripts/fetch-whisper-cpp-benchmark.sh source."
        }
    }
}

tasks.named("preBuild").configure {
    if (includeWhisperCppBenchmark.get()) {
        dependsOn(verifyWhisperCppBenchmarkSource)
    }
}

val graniteMetadataModel =
    graniteMetadataAssetRoot.file("granite-4.0-350m_q8_ekv1280.litertlm")
val verifyGraniteMetadataModel by tasks.registering {
    group = "verification"
    description = "Checks the pinned offline Granite title and summary model."
    inputs.file(graniteMetadataModel)

    doLast {
        val assetFile = graniteMetadataModel.asFile
        check(assetFile.isFile) {
            """
            Offline Granite metadata model is missing:
             - ${assetFile.relativeTo(projectDir)}

            Run scripts/fetch-granite-metadata-model.sh, then build again.
            """.trimIndent()
        }
        check(assetFile.length() == 468_209_584L) {
            "${assetFile.name} has ${assetFile.length()} bytes; expected 468209584."
        }
        val digest = MessageDigest.getInstance("SHA-256")
        assetFile.inputStream().buffered().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
        check(
            actualSha256 ==
                "c8e9a29493f62b7c44461fb36980987c4c1454c75e95f57ba0539a8edc9dce76",
        ) {
            "${assetFile.name} SHA-256 is $actualSha256; expected the pinned model."
        }
    }
}

val prepareGraniteMetadataProductionAssets by tasks.registering(Sync::class) {
    from(graniteMetadataAssetRoot) {
        into("granite")
    }
    into(graniteMetadataProductionAssetRoot)
}

tasks.named("preBuild").configure {
    dependsOn(verifyGraniteMetadataModel)
    dependsOn(prepareGraniteMetadataProductionAssets)
}

val verifyGraniteInDebugApk by tasks.registering {
    group = "verification"
    description = "Checks that the pinned Granite model is stored in the debug APK."
    dependsOn("packageDebug")

    val debugApk = layout.buildDirectory.file("outputs/apk/debug/app-debug.apk")
    inputs.file(debugApk)

    doLast {
        val apkFile = debugApk.get().asFile
        check(apkFile.isFile) {
            "Debug APK was not produced at ${apkFile.relativeTo(projectDir)}."
        }
        ZipFile(apkFile).use { apk ->
            val entryPath =
                "assets/granite/granite-4.0-350m_q8_ekv1280.litertlm"
            val entry = apk.getEntry(entryPath)
            checkNotNull(entry) {
                "The debug APK is missing the bundled Granite metadata model: $entryPath"
            }
            check(entry.size == 468_209_584L) {
                "The debug APK Granite model has ${entry.size} bytes; expected 468209584."
            }
            check(entry.method == ZipEntry.STORED) {
                "The debug APK Granite model must be stored uncompressed for AssetManager.openFd."
            }
        }
    }
}

tasks.configureEach {
    if (name == "assembleDebug") {
        dependsOn(verifyGraniteInDebugApk)
    }
}

val requiredParakeetAssets = mapOf(
    "parakeet_tdt_0.6b_v3_5s_i8_stateful.tflite" to (
        614_261_072L to "334745b8bc7fd372b1c213516f0b6338bb827b1a2abb3e77ad35fe6fea5cd16b"
        ),
    "tokenizer.json" to (
        1_159_960L to "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"
        ),
    "processor_config.json" to (
        392L to "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7"
        ),
    "config.json" to (
        1_153L to "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e"
        ),
)

val verifyParakeetAssets by tasks.registering {
    group = "verification"
    description = "Checks the pinned offline Parakeet TDT assets."
    inputs.files(requiredParakeetAssets.keys.map { parakeetSourceAssetRoot.file(it) })

    doLast {
        requiredParakeetAssets.forEach { (fileName, expected) ->
            val (expectedBytes, expectedSha256) = expected
            val assetFile = parakeetSourceAssetRoot.file(fileName).asFile
            check(assetFile.isFile) {
                "Missing Parakeet asset ${assetFile.relativeTo(projectDir)}. " +
                    "Run scripts/fetch-parakeet-assets.sh."
            }
            check(assetFile.length() == expectedBytes) {
                "$fileName has ${assetFile.length()} bytes; expected $expectedBytes."
            }
            val digest = MessageDigest.getInstance("SHA-256")
            assetFile.inputStream().buffered().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
            check(actualSha256 == expectedSha256) {
                "$fileName SHA-256 is $actualSha256; expected $expectedSha256."
            }
        }
    }
}

val prepareParakeetProductionAssets by tasks.registering(Sync::class) {
    dependsOn(verifyParakeetAssets)
    from(parakeetSourceAssetRoot)
    into(parakeetProductionAssetRoot.map { it.dir("parakeet") })
}

tasks.named("preBuild").configure {
    dependsOn(verifyParakeetAssets)
    if (useFullFloatDecoderProduction.get()) {
        dependsOn(prepareParakeetProductionAssets)
    }
}

val sileroVadAsset = sileroVadAssetRoot.file("silero/silero_vad.onnx")
val verifySileroVadAsset by tasks.registering {
    group = "verification"
    description = "Checks the pinned offline Silero VAD v6.2.1 model."
    inputs.file(sileroVadAsset)

    doLast {
        val assetFile = sileroVadAsset.asFile
        check(assetFile.isFile) {
            """
            Offline Silero VAD asset is missing:
             - ${assetFile.relativeTo(projectDir)}

            Run scripts/fetch-silero-vad-assets.sh, then build again.
            """.trimIndent()
        }
        check(assetFile.length() == 2_327_524L) {
            "silero_vad.onnx has ${assetFile.length()} bytes; expected 2327524. " +
                "Re-run scripts/fetch-silero-vad-assets.sh."
        }
        val digest = MessageDigest.getInstance("SHA-256")
        assetFile.inputStream().buffered().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
        check(
            actualSha256 ==
                "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        ) {
            "silero_vad.onnx SHA-256 is $actualSha256; expected the pinned v6.2.1 model."
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn(verifySileroVadAsset)
}

val verifyStrictMathGpuProbeAar by tasks.registering {
    group = "verification"
    description =
        "Verifies the opt-in generated LiteRT OpenCL strict-math diagnostic AAR."
    onlyIf { useStrictMathGpuProbe.get() }
    inputs.file(strictMathGpuProbeAar)
    inputs.property("expectedSha256", strictMathGpuProbeExpectedSha256)
    inputs.property("patchedAcceleratorSha256", strictMathGpuProbePatchedAcceleratorSha256)

    doLast {
        val aarFile = strictMathGpuProbeAar.asFile
        check(aarFile.isFile) {
            """
            Strict-math LiteRT diagnostic AAR is missing:
             - ${aarFile.relativeTo(rootProject.projectDir)}

            Run scripts/prepare-litert-strict-math-gpu-probe.sh, then rebuild
            with -PincludeDecoderSplitDiagnostics=true
            -PuseStrictMathGpuProbe=true.
            """.trimIndent()
        }
        check(aarFile.length() == strictMathGpuProbeExpectedBytes) {
            "Strict-math probe AAR has ${aarFile.length()} bytes; " +
                "expected $strictMathGpuProbeExpectedBytes."
        }
        val aarDigest = MessageDigest.getInstance("SHA-256")
        aarFile.inputStream().buffered().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                aarDigest.update(buffer, 0, count)
            }
        }
        val actualAarSha256 = aarDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualAarSha256 == strictMathGpuProbeExpectedSha256) {
            "Strict-math probe AAR SHA-256 is $actualAarSha256; " +
                "expected $strictMathGpuProbeExpectedSha256. Re-run " +
                "scripts/prepare-litert-strict-math-gpu-probe.sh."
        }

        ZipFile(aarFile).use { archive ->
            val acceleratorEntry = checkNotNull(
                archive.getEntry(strictMathGpuProbeAcceleratorEntry),
            ) {
                "Strict-math probe AAR is missing $strictMathGpuProbeAcceleratorEntry."
            }
            val acceleratorBytes = archive.getInputStream(acceleratorEntry).use { it.readBytes() }
            val acceleratorSha256 = MessageDigest.getInstance("SHA-256")
                .digest(acceleratorBytes)
                .joinToString("") { "%02x".format(it) }
            check(acceleratorSha256 == strictMathGpuProbePatchedAcceleratorSha256) {
                "Patched arm64 accelerator SHA-256 is $acceleratorSha256; " +
                    "expected $strictMathGpuProbePatchedAcceleratorSha256."
            }
            check(acceleratorBytes[strictMathGpuProbeFlagOffset] == 0.toByte()) {
                "Strict-math probe byte at 0x${strictMathGpuProbeFlagOffset.toString(16)} " +
                    "is not NUL."
            }
            check(
                acceleratorBytes.copyOfRange(
                    strictMathGpuProbeFlagOffset + 1,
                    strictMathGpuProbeFlagOffset + strictMathGpuProbeFlag.size,
                ).contentEquals(strictMathGpuProbeFlag.copyOfRange(1, strictMathGpuProbeFlag.size)),
            ) {
                "Strict-math probe flag suffix changed unexpectedly."
            }
            check(
                (0..acceleratorBytes.size - strictMathGpuProbeFlag.size).none { offset ->
                    strictMathGpuProbeFlag.indices.all { index ->
                        acceleratorBytes[offset + index] == strictMathGpuProbeFlag[index]
                    }
                },
            ) {
                "-cl-fast-relaxed-math remains in the patched arm64 accelerator."
            }
        }
    }
}

tasks.named("preBuild").configure {
    if (useStrictMathGpuProbe.get()) {
        dependsOn(verifyStrictMathGpuProbeAar)
    }
}

val verifyFp16Fp32AccumGpuProbeAar by tasks.registering {
    group = "verification"
    description =
        "Verifies the opt-in LiteRT FP16-storage/FP32-accumulator diagnostic AAR."
    onlyIf { useFp16Fp32AccumGpuProbe.get() }
    inputs.file(fp16Fp32AccumGpuProbeAar)
    inputs.property("expectedSha256", fp16Fp32AccumGpuProbeExpectedSha256)
    inputs.property(
        "acceleratorSha256",
        fp16Fp32AccumGpuProbeAcceleratorSha256,
    )

    doLast {
        val aarFile = fp16Fp32AccumGpuProbeAar.asFile
        check(aarFile.isFile) {
            """
            Mixed-precision LiteRT diagnostic AAR is missing:
             - ${aarFile.relativeTo(rootProject.projectDir)}

            Run scripts/prepare-litert-fp16-fp32-accum-gpu-probe.py, then
            rebuild with -PuseFp16Fp32AccumGpuProbe=true.
            """.trimIndent()
        }
        check(aarFile.length() == fp16Fp32AccumGpuProbeExpectedBytes) {
            "Mixed-precision probe AAR has ${aarFile.length()} bytes; " +
                "expected $fp16Fp32AccumGpuProbeExpectedBytes."
        }
        fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
            .digest(this)
            .joinToString("") { "%02x".format(it) }
        val actualAarSha256 = aarFile.readBytes().sha256()
        check(actualAarSha256 == fp16Fp32AccumGpuProbeExpectedSha256) {
            "Mixed-precision probe AAR SHA-256 is $actualAarSha256; expected " +
                "$fp16Fp32AccumGpuProbeExpectedSha256."
        }
        ZipFile(aarFile).use { archive ->
            val entry = checkNotNull(
                archive.getEntry(fp16Fp32AccumGpuProbeAcceleratorEntry),
            ) {
                "Mixed-precision probe AAR is missing " +
                    fp16Fp32AccumGpuProbeAcceleratorEntry
            }
            val accelerator = archive.getInputStream(entry).use { it.readBytes() }
            check(
                accelerator.sha256() ==
                    fp16Fp32AccumGpuProbeAcceleratorSha256,
            ) {
                "Mixed-precision accelerator SHA-256 changed."
            }
            check(
                accelerator.copyOfRange(
                    fp16Fp32AccumGpuProbeOffset,
                    fp16Fp32AccumGpuProbeOffset +
                        fp16Fp32AccumGpuProbeInstruction.size,
                ).contentEquals(fp16Fp32AccumGpuProbeInstruction),
            ) {
                "Mixed-precision accelerator does not contain the pinned " +
                    "`mov w10, #1` instruction."
            }
        }
    }
}

tasks.named("preBuild").configure {
    if (useFp16Fp32AccumGpuProbe.get()) {
        dependsOn(verifyFp16Fp32AccumGpuProbeAar)
    }
}

val decoderSplitDiagnosticAssets = mapOf(
    "whisper_large_v3_turbo_30s_i4_decode_hidden.tflite" to (
        755_273_648L to "a73a5d6acdd2e9271b7bced77ba784bad2c9716f4946cdc9145e0d70d97e1d0b"
        ),
    "whisper_large_v3_turbo_active_row_projection_i8.tflite" to (
        67_011_640L to "220e0fa46e2bdcf3d0396762fb1bd39df169615458107ed2af28184c78ebbb35"
        ),
)
val selectedDecoderSplitDiagnosticAssets = when (decoderSplitDiagnosticStage.get()) {
    "post-layernorm" -> decoderSplitDiagnosticAssets
    "pre-layernorm" -> mapOf(
        "whisper_large_v3_turbo_30s_i4_decode_pre_layer_norm.tflite" to (
            755_273_648L to
                "5006280fc6a6b2be0d029bd411c7dd2348de7afda9b7c757d2210397e14afe01"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    "after-position-add" -> mapOf(
        "whisper_large_v3_turbo_decode_shared_after_position_add.tflite" to (
            755_273_920L to
                "ee557aa9924a66f0b54f70c80759804b137e77dbaa9661d3b12bb5291615495d"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    "after-cross-attention" -> mapOf(
        "whisper_large_v3_turbo_decode_shared_after_cross_attention.tflite" to (
            755_273_928L to
                "bf71f075f3dffee2d7d233fffc0d77559e624941424a24933830e94386df3f5a"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    "after-1-block" -> mapOf(
        "whisper_large_v3_turbo_decode_shared_after_1_block.tflite" to (
            755_273_920L to
                "ed01a9e67f3a5991580bc7892ec9f747d63ed587bab02a6ef1b5c72366d4a6f3"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    "after-2-blocks" -> mapOf(
        "whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite" to (
            755_273_920L to
                "f5926743061104e49dd12f532fe3d35a3025f791bc1ec75c4e2a0b7eef136337"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    "after-3-blocks" -> mapOf(
        "whisper_large_v3_turbo_decode_shared_after_3_blocks.tflite" to (
            755_273_920L to
                "9f4deb00a29115753cbdb0a663de08c34eec2079f9c15c5f98ebb314cbacb3ab"
            ),
        "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite" to (
            67_023_504L to
                "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a"
            ),
    )
    else -> error("Unsupported decoder split diagnostic stage.")
}

val verifyDecoderSplitDiagnosticAssets by tasks.registering {
    group = "verification"
    description = "Verifies opt-in decoder-split models used only by device diagnostics."
    onlyIf { includeDecoderSplitDiagnostics.get() }
    inputs.property("decoderSplitDiagnosticStage", decoderSplitDiagnosticStage)
    inputs.files(
        selectedDecoderSplitDiagnosticAssets.keys.map(decoderSplitDiagnosticDirectory::file),
    )

    doLast {
        check(selectedDecoderSplitDiagnosticAssets.isNotEmpty()) {
            "No verified assets are registered for ${decoderSplitDiagnosticStage.get()}."
        }
        selectedDecoderSplitDiagnosticAssets.forEach { (fileName, expected) ->
            val (expectedBytes, expectedSha256) = expected
            val assetFile = decoderSplitDiagnosticDirectory.file(fileName).asFile
            check(assetFile.isFile) {
                """
                Decoder-split diagnostic asset is missing:
                 - ${assetFile.relativeTo(rootProject.projectDir)}

                Run scripts/generate-whisper-decoder-split.sh generate, then rebuild
                with -PincludeDecoderSplitDiagnostics=true.
                """.trimIndent()
            }
            check(assetFile.length() == expectedBytes) {
                "$fileName has ${assetFile.length()} bytes; expected $expectedBytes."
            }
            val digest = MessageDigest.getInstance("SHA-256")
            assetFile.inputStream().buffered().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
            check(actualSha256 == expectedSha256) {
                "$fileName SHA-256 is $actualSha256; expected $expectedSha256."
            }
        }
    }
}

val prepareDecoderSplitDiagnosticAssets by tasks.registering {
    onlyIf { includeDecoderSplitDiagnostics.get() }
    dependsOn(verifyDecoderSplitDiagnosticAssets)
    inputs.property("decoderSplitDiagnosticStage", decoderSplitDiagnosticStage)
    inputs.files(
        selectedDecoderSplitDiagnosticAssets.keys.map(decoderSplitDiagnosticDirectory::file),
    )
    outputs.dir(decoderSplitDiagnosticAssetRoot)

    doLast {
        val assetRoot = decoderSplitDiagnosticAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare decoder-split assets outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper-decoder-split")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        selectedDecoderSplitDiagnosticAssets.keys.forEach { fileName ->
            val source = decoderSplitDiagnosticDirectory.file(fileName).asFile.toPath()
            val destination = outputDirectory.resolve(fileName).toPath()
            try {
                Files.createLink(destination, source)
            } catch (_: UnsupportedOperationException) {
                Files.copy(source, destination)
            } catch (_: java.nio.file.FileSystemException) {
                Files.copy(source, destination)
            }
        }
    }
}

tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    if (includeDecoderSplitDiagnostics.get()) {
        dependsOn(prepareDecoderSplitDiagnosticAssets)
    }
}

val verifyRewrittenDecoderPrefixDiagnostic by tasks.registering {
    group = "verification"
    description = "Verifies the generated rewritten INT8 decoder-prefix diagnostic model."
    onlyIf { includeRewrittenDecoderPrefixDiagnostic.get() }
    inputs.file(rewrittenDecoderPrefixFile)
    inputs.file(rewrittenDecoderSoftmaxProbeFile)
    inputs.file(rewrittenDecoderMaskAddProbeFile)
    inputs.file(rewrittenDecoderScoreProbeFile)
    inputs.file(rewrittenDecoderValueProbeFile)
    inputs.file(rewrittenDecoderKProjectionProbeFile)
    inputs.property("rewrittenDecoderPrefixStage", rewrittenDecoderPrefixStage)
    inputs.property("rewrittenDecoderPrefixSha256", rewrittenDecoderPrefixSha256)
    inputs.property("rewrittenDecoderPrefixBytes", rewrittenDecoderPrefixBytes)

    doLast {
        val expectedSha256 = rewrittenDecoderPrefixSha256.get()
        check(expectedSha256.matches(Regex("[0-9a-f]{64}"))) {
            """
            -PrewrittenDecoderPrefixSha256 must be the exact SHA-256 printed by
            WhisperAttentionKeyTransposeTool generate-prefix.
            """.trimIndent()
        }
        val assetFile = rewrittenDecoderPrefixFile.asFile
        check(assetFile.isFile) {
            """
            Rewritten decoder-prefix diagnostic asset is missing:
             - ${assetFile.relativeTo(rootProject.projectDir)}

            Generate ${rewrittenDecoderPrefixStage.get()} with
            WhisperAttentionKeyTransposeTool generate-prefix, then rebuild.
            """.trimIndent()
        }
        val expectedBytes = rewrittenDecoderPrefixBytes.get()
        check(assetFile.length() == expectedBytes) {
            "decoder-prefix.tflite has ${assetFile.length()} bytes; expected $expectedBytes."
        }
        val digest = MessageDigest.getInstance("SHA-256")
        assetFile.inputStream().buffered().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actualSha256 = digest.digest().joinToString("") { "%02x".format(it) }
        check(actualSha256 == expectedSha256) {
            "decoder-prefix.tflite SHA-256 is $actualSha256; expected $expectedSha256."
        }
        val softmaxFile = rewrittenDecoderSoftmaxProbeFile.asFile
        check(softmaxFile.isFile && softmaxFile.length() == 552L) {
            "Generate the pinned isolated softmax probe before packaging diagnostics."
        }
        val softmaxDigest = MessageDigest.getInstance("SHA-256")
        softmaxFile.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                softmaxDigest.update(buffer, 0, count)
            }
        }
        val actualSoftmaxSha256 =
            softmaxDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualSoftmaxSha256 == rewrittenDecoderSoftmaxProbeSha256) {
            "softmax-probe.tflite SHA-256 is $actualSoftmaxSha256; " +
                "expected $rewrittenDecoderSoftmaxProbeSha256."
        }
        val maskAddFile = rewrittenDecoderMaskAddProbeFile.asFile
        check(maskAddFile.isFile && maskAddFile.length() == 664L) {
            "Generate the pinned isolated causal-mask ADD probe before packaging diagnostics."
        }
        val maskAddDigest = MessageDigest.getInstance("SHA-256")
        maskAddFile.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                maskAddDigest.update(buffer, 0, count)
            }
        }
        val actualMaskAddSha256 =
            maskAddDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualMaskAddSha256 == rewrittenDecoderMaskAddProbeSha256) {
            "mask-add-probe.tflite SHA-256 is $actualMaskAddSha256; " +
                "expected $rewrittenDecoderMaskAddProbeSha256."
        }
        val scoreFile = rewrittenDecoderScoreProbeFile.asFile
        check(scoreFile.isFile && scoreFile.length() == 620L) {
            "Generate the pinned isolated score-BMM probe before packaging diagnostics."
        }
        val scoreDigest = MessageDigest.getInstance("SHA-256")
        scoreFile.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                scoreDigest.update(buffer, 0, count)
            }
        }
        val actualScoreSha256 =
            scoreDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualScoreSha256 == rewrittenDecoderScoreProbeSha256) {
            "score-probe.tflite SHA-256 is $actualScoreSha256; " +
                "expected $rewrittenDecoderScoreProbeSha256."
        }
        val valueFile = rewrittenDecoderValueProbeFile.asFile
        check(valueFile.isFile && valueFile.length() == 664L) {
            "Generate the pinned isolated attention-value BMM probe before packaging diagnostics."
        }
        val valueDigest = MessageDigest.getInstance("SHA-256")
        valueFile.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                valueDigest.update(buffer, 0, count)
            }
        }
        val actualValueSha256 =
            valueDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualValueSha256 == rewrittenDecoderValueProbeSha256) {
            "value-probe.tflite SHA-256 is $actualValueSha256; " +
                "expected $rewrittenDecoderValueProbeSha256."
        }
        val kProjectionFile = rewrittenDecoderKProjectionProbeFile.asFile
        check(kProjectionFile.isFile && kProjectionFile.length() == 1_654_640L) {
            "Generate the pinned isolated K-projection probe before packaging diagnostics."
        }
        val kProjectionDigest = MessageDigest.getInstance("SHA-256")
        kProjectionFile.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                kProjectionDigest.update(buffer, 0, count)
            }
        }
        val actualKProjectionSha256 =
            kProjectionDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualKProjectionSha256 == rewrittenDecoderKProjectionProbeSha256) {
            "k-projection-probe.tflite SHA-256 is $actualKProjectionSha256; " +
                "expected $rewrittenDecoderKProjectionProbeSha256."
        }
    }
}

val prepareRewrittenDecoderPrefixDiagnostic by tasks.registering {
    onlyIf { includeRewrittenDecoderPrefixDiagnostic.get() }
    dependsOn(verifyRewrittenDecoderPrefixDiagnostic)
    inputs.file(rewrittenDecoderPrefixFile)
    inputs.file(rewrittenDecoderSoftmaxProbeFile)
    inputs.file(rewrittenDecoderMaskAddProbeFile)
    inputs.file(rewrittenDecoderScoreProbeFile)
    inputs.file(rewrittenDecoderValueProbeFile)
    inputs.file(rewrittenDecoderKProjectionProbeFile)
    inputs.property("rewrittenDecoderPrefixStage", rewrittenDecoderPrefixStage)
    inputs.property("rewrittenDecoderPrefixSha256", rewrittenDecoderPrefixSha256)
    inputs.property("rewrittenDecoderPrefixBytes", rewrittenDecoderPrefixBytes)
    outputs.dir(rewrittenDecoderPrefixAssetRoot)

    doLast {
        val assetRoot = rewrittenDecoderPrefixAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare the decoder-prefix asset outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper-decoder-prefix")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        val destination = outputDirectory.resolve("decoder-prefix.tflite").toPath()
        try {
            Files.createLink(destination, rewrittenDecoderPrefixFile.asFile.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.copy(rewrittenDecoderPrefixFile.asFile.toPath(), destination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(rewrittenDecoderPrefixFile.asFile.toPath(), destination)
        }
        val softmaxDestination = outputDirectory.resolve("softmax-probe.tflite").toPath()
        try {
            Files.createLink(
                softmaxDestination,
                rewrittenDecoderSoftmaxProbeFile.asFile.toPath(),
            )
        } catch (_: UnsupportedOperationException) {
            Files.copy(rewrittenDecoderSoftmaxProbeFile.asFile.toPath(), softmaxDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(rewrittenDecoderSoftmaxProbeFile.asFile.toPath(), softmaxDestination)
        }
        val maskAddDestination = outputDirectory.resolve("mask-add-probe.tflite").toPath()
        try {
            Files.createLink(
                maskAddDestination,
                rewrittenDecoderMaskAddProbeFile.asFile.toPath(),
            )
        } catch (_: UnsupportedOperationException) {
            Files.copy(rewrittenDecoderMaskAddProbeFile.asFile.toPath(), maskAddDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(rewrittenDecoderMaskAddProbeFile.asFile.toPath(), maskAddDestination)
        }
        val scoreDestination = outputDirectory.resolve("score-probe.tflite").toPath()
        try {
            Files.createLink(
                scoreDestination,
                rewrittenDecoderScoreProbeFile.asFile.toPath(),
            )
        } catch (_: UnsupportedOperationException) {
            Files.copy(rewrittenDecoderScoreProbeFile.asFile.toPath(), scoreDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(rewrittenDecoderScoreProbeFile.asFile.toPath(), scoreDestination)
        }
        val kProjectionDestination =
            outputDirectory.resolve("k-projection-probe.tflite").toPath()
        val valueDestination = outputDirectory.resolve("value-probe.tflite").toPath()
        try {
            Files.createLink(
                valueDestination,
                rewrittenDecoderValueProbeFile.asFile.toPath(),
            )
        } catch (_: UnsupportedOperationException) {
            Files.copy(rewrittenDecoderValueProbeFile.asFile.toPath(), valueDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(rewrittenDecoderValueProbeFile.asFile.toPath(), valueDestination)
        }
        try {
            Files.createLink(
                kProjectionDestination,
                rewrittenDecoderKProjectionProbeFile.asFile.toPath(),
            )
        } catch (_: UnsupportedOperationException) {
            Files.copy(
                rewrittenDecoderKProjectionProbeFile.asFile.toPath(),
                kProjectionDestination,
            )
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(
                rewrittenDecoderKProjectionProbeFile.asFile.toPath(),
                kProjectionDestination,
            )
        }
    }
}

tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    if (includeRewrittenDecoderPrefixDiagnostic.get()) {
        dependsOn(prepareRewrittenDecoderPrefixDiagnostic)
    }
}

val prepareInt8FcRegressionDiagnostic by tasks.registering {
    group = "verification"
    description = "Packages only the pinned Whisper INT8 K-projection regression model."
    onlyIf { includeInt8FcRegressionDiagnostic.get() }
    inputs.file(rewrittenDecoderKProjectionProbeFile)
    inputs.file(int8FcRegressionFixtureFile)
    inputs.property("expectedSha256", rewrittenDecoderKProjectionProbeSha256)
    inputs.property("expectedFixtureSha256", int8FcRegressionFixtureSha256)
    outputs.dir(int8FcRegressionAssetRoot)

    doLast {
        val source = rewrittenDecoderKProjectionProbeFile.asFile
        check(source.isFile && source.length() == 1_654_640L) {
            "Generate the pinned isolated K-projection probe before packaging diagnostics."
        }
        val digest = MessageDigest.getInstance("SHA-256")
        source.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actualSha256 =
            digest.digest().joinToString("") { "%02x".format(it) }
        check(actualSha256 == rewrittenDecoderKProjectionProbeSha256) {
            "k-projection-probe.tflite SHA-256 is $actualSha256; " +
                "expected $rewrittenDecoderKProjectionProbeSha256."
        }
        val fixture = int8FcRegressionFixtureFile.asFile
        check(fixture.isFile && fixture.length() == 655_360L) {
            "Generate the pinned Whisper INT8 FC step-7 fixture before packaging diagnostics."
        }
        val fixtureDigest = MessageDigest.getInstance("SHA-256")
        fixture.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                fixtureDigest.update(buffer, 0, count)
            }
        }
        val actualFixtureSha256 =
            fixtureDigest.digest().joinToString("") { "%02x".format(it) }
        check(actualFixtureSha256 == int8FcRegressionFixtureSha256) {
            "INT8 FC fixture SHA-256 is $actualFixtureSha256; " +
                "expected $int8FcRegressionFixtureSha256."
        }

        val assetRoot = int8FcRegressionAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare the INT8 FC regression asset outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper-decoder-prefix")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        val destination = outputDirectory.resolve("k-projection-probe.tflite").toPath()
        try {
            Files.createLink(destination, source.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.copy(source.toPath(), destination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(source.toPath(), destination)
        }
        val fixtureDestination =
            outputDirectory.resolve("k-projection-step-7-input.f32").toPath()
        try {
            Files.createLink(fixtureDestination, fixture.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.copy(fixture.toPath(), fixtureDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(fixture.toPath(), fixtureDestination)
        }
    }
}

tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    if (includeInt8FcRegressionDiagnostic.get()) {
        dependsOn(prepareInt8FcRegressionDiagnostic)
    }
}

val prepareInt8FcArithmeticDiagnostic by tasks.registering {
    group = "verification"
    description = "Verifies and packages the isolated INT8 FC arithmetic matrix."
    onlyIf { includeInt8FcArithmeticDiagnostic.get() }
    inputs.dir(int8FcArithmeticDirectory)
    inputs.file(int8FcRegressionFixtureFile)
    inputs.property("expectedManifestSha256", int8FcArithmeticManifestSha256)
    inputs.property("expectedFixtureSha256", int8FcRegressionFixtureSha256)
    outputs.dir(int8FcArithmeticAssetRoot)

    doLast {
        val sourceDirectory = int8FcArithmeticDirectory.asFile
        val manifest = sourceDirectory.resolve("matrix-manifest.tsv")
        check(manifest.isFile) {
            "Generate the pinned INT8 FC arithmetic matrix before packaging diagnostics."
        }
        fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        val actualManifestSha256 = sha256(manifest)
        check(actualManifestSha256 == int8FcArithmeticManifestSha256) {
            "INT8 FC arithmetic manifest SHA-256 is $actualManifestSha256; " +
                "expected $int8FcArithmeticManifestSha256."
        }
        val expectedNames = listOf(
            "original-per-channel",
            "original-explicit-zero-points",
            "requantized-per-tensor",
            "requantized-uniform-per-channel",
            "synthetic-diagonal-scale-ramp",
            "synthetic-accumulation-ladder",
            "float32-original-control",
        )
        expectedNames.forEach { name ->
            val model = sourceDirectory.resolve("$name.tflite")
            check(model.isFile && model.length() in 1_000_000L..7_000_000L) {
                "INT8 FC arithmetic model is missing or invalid: ${model.name}"
            }
            if (name != "float32-original-control") {
                check(sourceDirectory.resolve("$name.scales.f32").isFile) {
                    "INT8 FC scale sidecar is missing for $name."
                }
                check(sourceDirectory.resolve("$name.sums.i32").isFile) {
                    "INT8 FC row-sum sidecar is missing for $name."
                }
            }
        }
        val fixture = int8FcRegressionFixtureFile.asFile
        check(fixture.isFile && sha256(fixture) == int8FcRegressionFixtureSha256) {
            "Generate the pinned Whisper INT8 FC step-7 fixture before packaging diagnostics."
        }

        val assetRoot = int8FcArithmeticAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare INT8 FC arithmetic assets outside app/build."
        }
        assetRoot.deleteRecursively()
        val destination = assetRoot.resolve("whisper-int8-fc-matrix")
        check(destination.mkdirs()) {
            "Could not create ${destination.relativeTo(rootProject.projectDir)}."
        }
        sourceDirectory.listFiles().orEmpty().filter(File::isFile).forEach { source ->
            Files.copy(source.toPath(), destination.resolve(source.name).toPath())
        }
        Files.copy(
            fixture.toPath(),
            destination.resolve("real-step-7-input.f32").toPath(),
        )
    }
}

tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    if (includeInt8FcArithmeticDiagnostic.get()) {
        dependsOn(prepareInt8FcArithmeticDiagnostic)
    }
}

val verifySplitWhisperProduction by tasks.registering {
    group = "verification"
    description = "Verifies the production split Whisper encoder and decoder."
    onlyIf { useSplitWhisperProduction.get() }
    inputs.files(splitWhisperEncoderFile, splitWhisperDecoderFile)
    inputs.files(kvCrossCacheFile, kvStepFile, kvProjectionFile)
    inputs.property("encoderBytes", splitWhisperEncoderBytes)
    inputs.property("encoderSha256", splitWhisperEncoderSha256)
    inputs.property("decoderBytes", splitWhisperDecoderBytes)
    inputs.property("decoderSha256", splitWhisperDecoderSha256)
    inputs.property("kvCrossCacheSha256", kvCrossCacheSha256)
    inputs.property("kvStepSha256", kvStepSha256)
    inputs.property("kvProjectionSha256", kvProjectionSha256)

    doLast {
        listOf(
            Triple(splitWhisperEncoderFile.asFile, splitWhisperEncoderBytes, splitWhisperEncoderSha256),
            Triple(splitWhisperDecoderFile.asFile, splitWhisperDecoderBytes, splitWhisperDecoderSha256),
            Triple(kvCrossCacheFile.asFile, kvCrossCacheBytes, kvCrossCacheSha256),
            Triple(kvStepFile.asFile, kvStepBytes, kvStepSha256),
            Triple(kvProjectionFile.asFile, kvProjectionBytes, kvProjectionSha256),
        ).forEach { (model, expectedBytes, expectedSha256) ->
            check(model.isFile) { "Split Whisper model is missing: $model" }
            check(model.length() == expectedBytes) {
                "${model.name} has ${model.length()} bytes; expected $expectedBytes."
            }
            val digest = MessageDigest.getInstance("SHA-256")
            model.inputStream().buffered().use { stream ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            check(actual == expectedSha256) {
                "${model.name} SHA-256 is $actual; expected $expectedSha256."
            }
        }
    }
}

val prepareSplitWhisperProduction by tasks.registering {
    onlyIf { useSplitWhisperProduction.get() }
    dependsOn(verifyWhisperAssets, verifySplitWhisperProduction)
    inputs.files(splitWhisperEncoderFile, splitWhisperDecoderFile)
    inputs.files(kvCrossCacheFile, kvStepFile, kvProjectionFile)
    inputs.files(
        listOf(
            "tokenizer.json",
            "generation_config.json",
            "preprocessor_config.json",
        ).map(whisperAssetDirectory::file),
    )
    outputs.dir(splitWhisperProductionAssetRoot)

    doLast {
        val assetRoot = splitWhisperProductionAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare split production assets outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        listOf(
            splitWhisperEncoderFile.asFile to outputDirectory.resolve(
                "whisper_large_v3_turbo_split_encoder.tflite",
            ),
            splitWhisperDecoderFile.asFile to outputDirectory.resolve(
                "whisper_large_v3_turbo_split_decoder.tflite",
            ),
            kvCrossCacheFile.asFile to outputDirectory.resolve(
                "whisper_large_v3_turbo_kv_cross_cache.tflite",
            ),
            kvStepFile.asFile to outputDirectory.resolve(
                "whisper_large_v3_turbo_kv_step.tflite",
            ),
            kvProjectionFile.asFile to outputDirectory.resolve(
                "whisper_large_v3_turbo_kv_projection_fp32.tflite",
            ),
        ).forEach { (source, destination) ->
            try {
                Files.createLink(destination.toPath(), source.toPath())
            } catch (_: UnsupportedOperationException) {
                Files.copy(source.toPath(), destination.toPath())
            } catch (_: java.nio.file.FileSystemException) {
                Files.copy(source.toPath(), destination.toPath())
            }
        }
        listOf(
            "tokenizer.json",
            "generation_config.json",
            "preprocessor_config.json",
        ).forEach { fileName ->
            Files.copy(
                whisperAssetDirectory.file(fileName).asFile.toPath(),
                outputDirectory.resolve(fileName).toPath(),
            )
        }
    }
}

val verifyFullFloatDecoderDiagnostic by tasks.registering {
    group = "verification"
    description = "Verifies the fail-closed complete FLOAT32 decoder GPU model."
    onlyIf {
        includeFullFloatDecoderDiagnostic.get() ||
            useFullFloatDecoderProduction.get()
    }
    inputs.file(fullFloatDecoderDiagnosticFile)
    inputs.property("expectedBytes", fullFloatDecoderDiagnosticBytes)
    inputs.property("expectedSha256", fullFloatDecoderDiagnosticSha256)

    doLast {
        val model = fullFloatDecoderDiagnosticFile.asFile
        check(model.isFile) {
            """
            Complete FLOAT32 decoder diagnostic model is missing:
             - ${model.relativeTo(rootProject.projectDir)}

            Generate it with WhisperAttentionKeyTransposeTool
            generate-full-float-decoder, then rebuild.
            """.trimIndent()
        }
        check(model.length() == fullFloatDecoderDiagnosticBytes) {
            "Full FLOAT32 decoder has ${model.length()} bytes; " +
                "expected $fullFloatDecoderDiagnosticBytes."
        }
        val digest = MessageDigest.getInstance("SHA-256")
        model.inputStream().buffered().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        check(actual == fullFloatDecoderDiagnosticSha256) {
            "Full FLOAT32 decoder SHA-256 is $actual; " +
                "expected $fullFloatDecoderDiagnosticSha256."
        }
    }
}

val prepareFullFloatDecoderProduction by tasks.registering {
    onlyIf { useFullFloatDecoderProduction.get() }
    dependsOn(verifyWhisperAssets, verifyFullFloatDecoderDiagnostic)
    inputs.file(fullFloatDecoderDiagnosticFile)
    inputs.files(
        listOf(
            "tokenizer.json",
            "generation_config.json",
            "preprocessor_config.json",
        ).map(whisperAssetDirectory::file),
    )
    outputs.dir(fullFloatDecoderProductionAssetRoot)

    doLast {
        val assetRoot = fullFloatDecoderProductionAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare production GPU assets outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        val modelDestination =
            outputDirectory.resolve("whisper_large_v3_turbo_full_float_decoder.tflite").toPath()
        try {
            Files.createLink(modelDestination, fullFloatDecoderDiagnosticFile.asFile.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.copy(fullFloatDecoderDiagnosticFile.asFile.toPath(), modelDestination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(fullFloatDecoderDiagnosticFile.asFile.toPath(), modelDestination)
        }
        listOf(
            "tokenizer.json",
            "generation_config.json",
            "preprocessor_config.json",
        ).forEach { fileName ->
            Files.copy(
                whisperAssetDirectory.file(fileName).asFile.toPath(),
                outputDirectory.resolve(fileName).toPath(),
            )
        }
    }
}

tasks.matching {
    it.name.startsWith("merge") &&
        it.name.endsWith("Assets") &&
        !it.name.contains("AndroidTest")
}.configureEach {
    if (useFullFloatDecoderProduction.get()) {
        dependsOn(prepareFullFloatDecoderProduction)
    } else if (useSplitWhisperProduction.get()) {
        dependsOn(prepareSplitWhisperProduction)
    }
}

// AGP's lint pipeline reads source-set assets directly instead of going
// through the merge tasks above. Its task names include both lint... and
// generate...Lint..., so wire every lint-related task to the asset producer.
tasks.matching { it.name.contains("lint", ignoreCase = true) }.configureEach {
    if (useFullFloatDecoderProduction.get()) {
        dependsOn(prepareFullFloatDecoderProduction)
    } else if (useSplitWhisperProduction.get()) {
        dependsOn(prepareSplitWhisperProduction)
    }
}

val prepareFullFloatDecoderDiagnostic by tasks.registering {
    onlyIf { includeFullFloatDecoderDiagnostic.get() }
    dependsOn(verifyFullFloatDecoderDiagnostic)
    inputs.file(fullFloatDecoderDiagnosticFile)
    outputs.dir(fullFloatDecoderDiagnosticAssetRoot)

    doLast {
        val assetRoot = fullFloatDecoderDiagnosticAssetRoot.get().asFile
        check(assetRoot.toPath().startsWith(layout.buildDirectory.get().asFile.toPath())) {
            "Refusing to prepare the full decoder asset outside app/build."
        }
        assetRoot.deleteRecursively()
        val outputDirectory = assetRoot.resolve("whisper-full-float-decoder")
        check(outputDirectory.mkdirs()) {
            "Could not create ${outputDirectory.relativeTo(rootProject.projectDir)}."
        }
        val destination = outputDirectory.resolve("decoder.tflite").toPath()
        try {
            Files.createLink(destination, fullFloatDecoderDiagnosticFile.asFile.toPath())
        } catch (_: UnsupportedOperationException) {
            Files.copy(fullFloatDecoderDiagnosticFile.asFile.toPath(), destination)
        } catch (_: java.nio.file.FileSystemException) {
            Files.copy(fullFloatDecoderDiagnosticFile.asFile.toPath(), destination)
        }
    }
}

tasks.matching { it.name == "mergeDebugAndroidTestAssets" }.configureEach {
    if (includeFullFloatDecoderDiagnostic.get()) {
        dependsOn(prepareFullFloatDecoderDiagnostic)
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.06.00")

    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.activity:activity-compose:1.12.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.4")

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.material3:material3:1.4.0")

    implementation("androidx.room:room-runtime:2.8.4")
    implementation("androidx.room:room-ktx:2.8.4")
    kapt("androidx.room:room-compiler:2.8.4")

    implementation("androidx.work:work-runtime-ktx:2.11.2")
    implementation("androidx.media3:media3-exoplayer:1.10.1")
    implementation("androidx.media3:media3-session:1.10.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.14.0")
    if (useStrictMathGpuProbe.get() || useFp16Fp32AccumGpuProbe.get()) {
        // Diagnostic-only: a generated local copy changes one OpenCL compiler
        // option or precision-selection instruction in the arm64 accelerator.
        // A file dependency has no POM, so retain LiteRT's public API explicitly.
        implementation(
            files(
                if (useFp16Fp32AccumGpuProbe.get()) {
                    fp16Fp32AccumGpuProbeAar
                } else {
                    strictMathGpuProbeAar
                },
            ),
        )
        implementation("com.google.ai.edge.litert:litert-api:2.1.6") {
            exclude(group = "com.google.android.play", module = "ai-delivery")
        }
    } else {
        implementation("com.google.ai.edge.litert:litert:2.1.6") {
            // This app bundles its model. LiteRT's optional Play AI delivery
            // integration would add Play Asset Delivery components and
            // ACCESS_NETWORK_STATE even though the CPU Interpreter does not use
            // them.
            exclude(group = "com.google.android.play", module = "ai-delivery")
        }
    }
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
    testImplementation("androidx.room:room-testing:2.8.4")

    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.room:room-testing:2.8.4")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
