# Standalone native ASR runtime. Assets are pinned build inputs, never downloads.
set(whisper_assets "${CMAKE_CURRENT_BINARY_DIR}/generated/whisper_assets.cpp")
add_custom_command(OUTPUT "${whisper_assets}"
  COMMAND "${CMAKE_COMMAND}" "-DROOT=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DOUTPUT=${whisper_assets}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedWhisper.cmake"
  DEPENDS resources/whisper/multilingual.tiktoken resources/whisper/gpt2.tiktoken
    resources/whisper/profiles.json cmake/EmbedWhisper.cmake VERBATIM)
add_library(slopfab_whisper STATIC src/whisper/frontend.cpp src/whisper/tokenizer.cpp
  src/whisper/model.cpp src/whisper/cpu.cpp src/whisper/transcribe.cpp "${whisper_assets}")
target_link_libraries(slopfab_whisper PUBLIC slopfab_core)
target_include_directories(slopfab_whisper PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/whisper")
set_target_properties(slopfab_whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET slopfab_cuda)
  target_sources(slopfab_cuda PRIVATE src/whisper/cuda.cu)
  target_link_libraries(slopfab_whisper PUBLIC slopfab_cuda)
endif()
if(TARGET slopfab_vulkan)
  target_sources(slopfab_vulkan PRIVATE src/whisper/vulkan.cpp)
  target_link_libraries(slopfab_whisper PUBLIC slopfab_vulkan)
endif()
if(MSVC)
  target_compile_options(slopfab_whisper PRIVATE /utf-8 /EHsc)
endif()
