# --- vulkan -----------------------------------------------------------------

# Vulkan is independent of CUDA and implements the core frame-converter
# interface. The Khronos core ABI headers are pinned in-tree, while the
# platform loader is opened dynamically by runtime.cpp. Consequently enabling
# this target needs no Vulkan SDK at build time or Vulkan loader at process
# startup.
if(SLOPFAB_ENABLE_VULKAN)
  include(cmake/Shaders.cmake)
  include(cmake/ShaderManifest.cmake)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
  configure_file(src/vulkan/embedded_yuv_spv.h.in
    generated/vulkan/embedded_yuv_spv.h @ONLY)
  configure_file(src/vulkan/embedded_tensor_spv.h.in
    generated/vulkan/embedded_tensor_spv.h @ONLY)

  add_library(slopfab_vulkan STATIC
    src/vulkan/runtime.cpp
    src/vulkan/runtime_buffer.cpp
    src/vulkan/runtime_pipeline.cpp
    src/vulkan/runtime_compute.cpp
    src/vulkan/tensor.cpp
    src/vulkan/tensor_pipelines.cpp
    src/vulkan/tensor_weights.cpp
    src/vulkan/tensor_pointwise.cpp
    src/vulkan/tensor_norm.cpp
    src/vulkan/tensor_dit.cpp
    src/vulkan/tensor_conditioning.cpp
    src/vulkan/tensor_reference.cpp
    src/vulkan/tensor_audio.cpp
    src/vulkan/tensor_gemm.cpp
    src/vulkan/tensor_attention.cpp
    src/vulkan/text_layer.cpp
    src/vulkan/text_encoder.cpp
    src/vulkan/vision_stage.cpp
    src/vulkan/lora.cpp
    src/vulkan/dit_block.cpp
    src/vulkan/dit_denoise.cpp
    src/vulkan/dit_graph.cpp
    src/vulkan/dit_transformer.cpp
    src/vulkan/audio_decoder.cpp
    src/vulkan/keyframe_encoder.cpp
    src/vulkan/reference_encoder.cpp
    src/vulkan/vae_decoder.cpp
    src/vulkan/vae_vit_block.cpp
    src/vulkan/yuv_converter.cpp)
  target_include_directories(slopfab_vulkan
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan)
  target_link_libraries(slopfab_vulkan PUBLIC slopfab_core)
  target_compile_definitions(slopfab_vulkan PRIVATE VK_NO_PROTOTYPES)
  if(CMAKE_DL_LIBS)
    target_link_libraries(slopfab_vulkan PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(MSVC)
    target_compile_options(slopfab_vulkan PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_vulkan PRIVATE -Wall -Wextra)
  endif()
endif()

