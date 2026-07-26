plugins {
    application
}

if (!JavaVersion.current().isCompatibleWith(JavaVersion.VERSION_17)) {
    throw GradleException(
        "Whisper decoder-split tooling requires JDK 17 or newer. " +
            "Use scripts/generate-whisper-decoder-split.sh, or set JAVA_HOME " +
            "to a JDK 17+ installation before invoking this build directly.",
    )
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite-metadata:0.2.0") {
        isTransitive = false
    }
    implementation("com.google.flatbuffers:flatbuffers-java:1.12.0") {
        isTransitive = false
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

tasks.withType<org.gradle.api.tasks.compile.JavaCompile>().configureEach {
    options.release.set(17)
}

application {
    applicationName = "whisper-decoder-split"
    mainClass = "com.example.voicememo.tools.WhisperDecoderSplitTool"
    applicationDefaultJvmArgs = listOf("-Xms128m", "-Xmx1g")
}

val hostSelfTest by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Runs content-free host checks for the decoder-split tool."
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass = application.mainClass
    args("self-test")
}

val floatDecoderProbeSelfTest by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Runs content-free checks for the FLOAT32 decoder control tool."
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass = "com.example.voicememo.tools.WhisperFloatDecoderProbeTool"
    args("self-test")
}

tasks.named("check") {
    dependsOn(hostSelfTest)
    dependsOn(floatDecoderProbeSelfTest)
}
