# --- core -------------------------------------------------------------------

add_library(slopfab_core STATIC
  src/krea/plan.cpp
  src/fibo/plan.cpp
  src/fibo/render.cpp
  src/music/plan.cpp
  src/music/tokenizer.cpp
  src/core/sampling_effect.cpp
  src/core/lcs.cpp
  src/seedvr2/plan.cpp
  src/core/json.cpp
  src/core/nf4.cpp
  src/core/w4a8.cpp
  src/core/int8_weight.cpp
  src/core/lora.cpp
  src/core/sampling_settings.cpp
  src/core/lora_grid.cpp
  src/core/safetensors.cpp
  src/core/tensor_convert.cpp
  src/core/device_tensor.cpp
  src/core/model_geometry.cpp
  src/core/model_profile.cpp
  src/core/image.cpp
  src/core/upscale.cpp
  src/core/latent_upscale.cpp
  src/core/inpaint.cpp
  src/core/reference_media.cpp
  src/core/refmod.cpp
  src/core/continuation.cpp
  src/core/reference_conditioning.cpp
  src/core/image_platform.cpp
  src/core/safetensors_write.cpp
  src/core/sha256.cpp
  src/dit/adaln.cpp
  src/dit/h3_lora.cpp
  src/dit/hyperflow.cpp
  src/core/conditioning_settings.cpp
  src/conditioning_plan.cpp
  src/core/remake.cpp
  src/generation_validation.cpp
  src/dit/block_cache.cpp
  src/dit/chunking.cpp
  src/dit/checkpoint.cpp
  src/dit/pdd.cpp
  src/dit/vsa.cpp
  src/dit/veda.cpp
  src/dit/packing.cpp
  src/dit/ref2va.cpp
  src/dit/rope.cpp
  src/dit/step_cache.cpp
  src/dit/motion_cache.cpp
  src/vae/keyframe_encoder.cpp
  src/vae/audio_primitives.cpp
  src/vae/vit_block.cpp
  src/vae/sol_h3.cpp
  src/vae/decode_pipeline.cpp
  src/pipeline.cpp
  src/sampling_plan.cpp
  src/sampler/noise.cpp
  src/sampler/scheduler.cpp
  src/sampler/er_sde.cpp
  src/sampler/beta.cpp
  src/text/tokenizer.cpp
  src/text/nfc.cpp
  src/text/tokenizer_contract.cpp
  src/text/encoder.cpp
  src/text/clip_projection.cpp
  src/text/prompt_embedding.cpp
  src/text/layer_capture.cpp
  src/text/qwen_vision.cpp
  src/video/y4m.cpp
  src/video/y4m_compare.cpp
  src/audio/wav.cpp
)

target_sources(slopfab_core PRIVATE src/video/mux_disabled.cpp)
target_compile_definitions(slopfab_core PUBLIC SLOPFAB_WITH_FFMPEG=0)

target_include_directories(slopfab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The platform image decoder. Compiled in every configuration — so the two
# decoders can be compared and the code cannot rot unbuilt — which is why
# these are linked whether or not FFmpeg is on. Both ship with Windows.
if(WIN32)
  target_link_libraries(slopfab_core PRIVATE windowscodecs ole32 bcrypt gdi32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(PNG REQUIRED)
  find_package(JPEG REQUIRED)
  target_link_libraries(slopfab_core PRIVATE PNG::PNG JPEG::JPEG)
endif()

# The RGB -> YUV 4:2:0 conversion in src/video/y4m.cpp is threaded. MSVC needs
# nothing for std::thread, but every other toolchain wants -pthread both to
# compile and to link, and without this the non-MSVC build breaks. Requested
# here as well as in the CUDA block below because slopfab_core must build on its
# own, on a machine with no CUDA toolkit at all.
find_package(Threads REQUIRED)
target_link_libraries(slopfab_core PUBLIC Threads::Threads)

if(MSVC)
  # /utf-8 keeps string literals well-defined regardless of the system codepage.
  target_compile_options(slopfab_core PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(slopfab_core PRIVATE -Wall -Wextra)
endif()

