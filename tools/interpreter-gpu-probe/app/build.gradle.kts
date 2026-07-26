import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.voicememo.interpretergpuprobe"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.voicememo.interpretergpuprobe"
        minSdk = 23
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs.useLegacyPackaging = false
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // LiteRT 2.1.x intentionally exposes a CPU-only Interpreter Maven API.
    // The last separately published standard GPU delegate is 1.4.2, so this
    // isolated package keeps its Java/native runtime out of Voice Memo.
    implementation("com.google.ai.edge.litert:litert:1.4.2") {
        exclude(group = "com.google.android.play", module = "ai-delivery")
    }
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2") {
        exclude(group = "com.google.android.play", module = "ai-delivery")
    }
    implementation("com.google.ai.edge.litert:litert-gpu-api:1.4.2") {
        exclude(group = "com.google.android.play", module = "ai-delivery")
    }

    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
