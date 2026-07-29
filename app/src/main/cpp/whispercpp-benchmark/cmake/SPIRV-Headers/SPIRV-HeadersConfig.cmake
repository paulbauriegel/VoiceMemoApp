if(NOT CMAKE_ANDROID_NDK)
    message(FATAL_ERROR "VoiceMemo's SPIRV-Headers package is Android-only.")
endif()

set(
    _voicememo_spirv_headers
    "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include"
)
if(NOT EXISTS "${_voicememo_spirv_headers}/spirv/unified1/spirv.hpp")
    message(FATAL_ERROR "Android NDK SPIR-V headers are missing.")
endif()

if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
    set_target_properties(
        SPIRV-Headers::SPIRV-Headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_voicememo_spirv_headers}"
    )
endif()
