cmake_minimum_required(VERSION 3.22.1)

project(voicememo_whispercpp_benchmark LANGUAGES C CXX)

if(NOT DEFINED WHISPER_CPP_SOURCE_DIR)
    message(FATAL_ERROR "WHISPER_CPP_SOURCE_DIR must point to the pinned whisper.cpp checkout")
endif()

if(NOT EXISTS "${WHISPER_CPP_SOURCE_DIR}/include/whisper.h")
    message(FATAL_ERROR "Pinned whisper.cpp sources are missing: ${WHISPER_CPP_SOURCE_DIR}")
endif()
if(NOT DEFINED VULKAN_HEADERS_DIR OR
   NOT EXISTS "${VULKAN_HEADERS_DIR}/include/vulkan/vulkan.hpp")
    message(FATAL_ERROR "VULKAN_HEADERS_DIR must point to the pinned Vulkan-Headers checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

if(ANDROID)
    set(
        SPIRV-Headers_DIR
        "${CMAKE_CURRENT_LIST_DIR}/cmake/SPIRV-Headers"
        CACHE PATH
        "Android NDK SPIR-V headers package"
        FORCE
    )
    unset(VOICEMEMO_HOST_C_COMPILER CACHE)
    unset(VOICEMEMO_HOST_CXX_COMPILER CACHE)
    find_program(
        VOICEMEMO_GLSLC
        NAMES glslc
        HINTS
            "${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64"
            "${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64"
            "${CMAKE_ANDROID_NDK}/shader-tools/windows-x86_64"
        NO_CMAKE_FIND_ROOT_PATH
        REQUIRED
    )
    set(
        Vulkan_GLSLC_EXECUTABLE
        "${VOICEMEMO_GLSLC}"
        CACHE FILEPATH
        "Host glslc used to compile embedded whisper.cpp Vulkan shaders"
        FORCE
    )
    find_program(
        VOICEMEMO_HOST_C_COMPILER
        NAMES clang
        HINTS
            "/usr/bin"
            "/Library/Developer/CommandLineTools/usr/bin"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
        REQUIRED
    )
    find_program(
        VOICEMEMO_HOST_CXX_COMPILER
        NAMES clang++
        HINTS
            "/usr/bin"
            "/Library/Developer/CommandLineTools/usr/bin"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH
        REQUIRED
    )
    set(VOICEMEMO_HOST_NINJA "${CMAKE_MAKE_PROGRAM}")
    set(
        VOICEMEMO_VULKAN_HOST_TOOLCHAIN
        "${CMAKE_CURRENT_BINARY_DIR}/voicememo-vulkan-host-toolchain.cmake"
    )
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/vulkan-host-toolchain.cmake.in"
        "${VOICEMEMO_VULKAN_HOST_TOOLCHAIN}"
        @ONLY
    )
    set(
        GGML_VULKAN_SHADERS_GEN_TOOLCHAIN
        "${VOICEMEMO_VULKAN_HOST_TOOLCHAIN}"
        CACHE FILEPATH
        "Host toolchain for whisper.cpp's Vulkan shader generator"
        FORCE
    )

    set(
        _voicememo_vulkan_cmake
        "${WHISPER_CPP_SOURCE_DIR}/ggml/src/ggml-vulkan/CMakeLists.txt"
    )
    file(READ "${_voicememo_vulkan_cmake}" _voicememo_vulkan_cmake_contents)
    set(
        _voicememo_ninja_argument
        "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    )
    if(NOT _voicememo_vulkan_cmake_contents MATCHES "-DCMAKE_MAKE_PROGRAM=")
        set(_voicememo_external_project_anchor "CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=")
        string(
            FIND
            "${_voicememo_vulkan_cmake_contents}"
            "${_voicememo_external_project_anchor}"
            _voicememo_external_project_anchor_index
        )
        if(_voicememo_external_project_anchor_index EQUAL -1)
            message(FATAL_ERROR "Pinned whisper.cpp Vulkan CMake layout changed.")
        endif()
        string(
            REPLACE
            "${_voicememo_external_project_anchor}"
            "CMAKE_ARGS ${_voicememo_ninja_argument}\n                   -DCMAKE_INSTALL_PREFIX="
            _voicememo_vulkan_cmake_contents
            "${_voicememo_vulkan_cmake_contents}"
        )
        file(WRITE "${_voicememo_vulkan_cmake}" "${_voicememo_vulkan_cmake_contents}")
    endif()

    set(
        _voicememo_vulkan_source
        "${WHISPER_CPP_SOURCE_DIR}/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
    )
    file(READ "${_voicememo_vulkan_source}" _voicememo_vulkan_source_contents)
    if(_voicememo_vulkan_source_contents MATCHES "[^.]vkGetPhysicalDeviceFeatures2\\(")
        string(
            REPLACE
            "vkGetPhysicalDeviceFeatures2("
            "ggml_vk_default_dispatcher().vkGetPhysicalDeviceFeatures2("
            _voicememo_vulkan_source_contents
            "${_voicememo_vulkan_source_contents}"
        )
        file(WRITE "${_voicememo_vulkan_source}" "${_voicememo_vulkan_source_contents}")
    endif()

    set(
        _voicememo_descriptor_capacity_anchor
        "    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());"
    )
    if(NOT _voicememo_vulkan_source_contents MATCHES "ggml_vk_descriptor_set_requirements_after_request\\([\n\r ]*ctx->pipeline_descriptor_set_requirements,[\n\r ]*ctx->descriptor_set_idx,[\n\r ]*1\\)")
        string(
            FIND
            "${_voicememo_vulkan_source_contents}"
            "${_voicememo_descriptor_capacity_anchor}"
            _voicememo_descriptor_capacity_anchor_index
        )
        if(_voicememo_descriptor_capacity_anchor_index EQUAL -1)
            message(FATAL_ERROR "Pinned whisper.cpp descriptor dispatch layout changed.")
        endif()
        string(
            REPLACE
            "${_voicememo_descriptor_capacity_anchor}"
            "    if (ctx->descriptor_set_idx >= ctx->descriptor_sets.size()) {\n        ctx->pipeline_descriptor_set_requirements =\n            ggml_vk_descriptor_set_requirements_after_request(\n                ctx->pipeline_descriptor_set_requirements,\n                ctx->descriptor_set_idx,\n                1);\n        ggml_pipeline_allocate_descriptor_sets(ctx);\n    }\n${_voicememo_descriptor_capacity_anchor}"
            _voicememo_vulkan_source_contents
            "${_voicememo_vulkan_source_contents}"
        )
        file(WRITE "${_voicememo_vulkan_source}" "${_voicememo_vulkan_source_contents}")
    endif()

endif()

add_subdirectory(
    "${WHISPER_CPP_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/whisper.cpp"
    EXCLUDE_FROM_ALL
)
target_include_directories(
    ggml-vulkan
    PRIVATE
    "${VULKAN_HEADERS_DIR}/include"
    "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include"
)

add_library(
    voicememo_whispercpp_benchmark
    SHARED
    whispercpp_benchmark_jni.cpp
)

target_compile_features(voicememo_whispercpp_benchmark PRIVATE cxx_std_17)
target_compile_definitions(
    voicememo_whispercpp_benchmark
    PRIVATE
    VOICEMEMO_WHISPER_CPP_REVISION="${WHISPER_CPP_REVISION}"
)
target_link_libraries(
    voicememo_whispercpp_benchmark
    PRIVATE
    whisper
    ggml
    android
    log
)
