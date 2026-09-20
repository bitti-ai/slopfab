# --- tests ------------------------------------------------------------------
if(SLOPFAB_BUILD_TESTS)
  enable_testing()
  include(cmake/TestCoverage.cmake)
  # Host tests. Each area contributes its own translation unit and registers
  # its cases with the shared harness, so adding tests never touches a file
  # someone else is editing.
  add_executable(slopfab_tests
    tests/harness.cpp
    tests/test_main.cpp
    tests/test_adaln.cpp
    tests/test_offload.cpp
    tests/test_audio_primitives.cpp
    tests/test_chunking.cpp
    tests/test_continuation.cpp
    tests/test_compare_metrics.cpp
    tests/test_decode_pipeline.cpp
    tests/test_stillprobe.cpp
    tests/test_image.cpp
    tests/test_reference_media.cpp
    tests/test_refmod.cpp
    tests/test_noise.cpp
    tests/test_packing.cpp
    tests/test_ref2va.cpp
    tests/test_vsa.cpp
    tests/test_safetensors.cpp
    tests/test_pipeline.cpp
    tests/test_model_descriptor.cpp
    tests/test_attention_descriptor.cpp
    tests/test_prompt_embedding.cpp
    tests/test_lora.cpp
    tests/test_sampler.cpp
    tests/test_sampling_settings.cpp
    tests/test_conditioning_settings.cpp
    tests/test_sampling_plan.cpp
    tests/test_step_cache.cpp
    tests/test_motion_cache.cpp
    tests/test_block_cache.cpp
    tests/test_tile_merge.cpp
    tests/test_tokenizer.cpp
    tests/test_tokenizer_json.cpp
    tests/test_qwen_vision.cpp
    tests/test_output.cpp
    tests/test_device_tensor.cpp
  )
  target_link_libraries(slopfab_tests PRIVATE slopfab_core)
  if(MSVC)
    target_compile_options(slopfab_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(slopfab_tests PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME unit COMMAND slopfab_tests)
  set_tests_properties(unit PROPERTIES LABELS synthetic SKIP_RETURN_CODE 77)
  if(SLOPFAB_WITH_FFMPEG)
    find_program(SLOPFAB_TEST_FFMPEG ffmpeg HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
    find_program(SLOPFAB_TEST_FFPROBE ffprobe HINTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin")
    if(SLOPFAB_TEST_FFMPEG AND SLOPFAB_TEST_FFPROBE)
      add_executable(slopfab_reference_decode_tests tests/test_reference_decode.cpp src/cli/reference_decode.cpp)
      target_link_libraries(slopfab_reference_decode_tests PRIVATE slopfab_core)
      add_test(NAME reference_media_decode COMMAND ${CMAKE_COMMAND}
        "-DPROBE=$<TARGET_FILE:slopfab_reference_decode_tests>" "-DFFMPEG=${SLOPFAB_TEST_FFMPEG}"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/reference_media_fixture"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reference_decode.cmake")
      set_tests_properties(reference_media_decode PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 60)
    endif()
  endif()

  if(SLOPFAB_ENABLE_VULKAN)
    add_executable(slopfab_vulkan_tests tests/harness.cpp tests/test_vulkan.cpp
      tests/test_vulkan_pipeline_sets.cpp
      tests/test_vulkan_conditioner_synthetic.cpp
      tests/test_vulkan_conditioner_checkpoint.cpp
      tests/test_vulkan_conditioner_integration.cpp
      tests/test_vulkan_dit_synthetic_1.cpp
      tests/test_vulkan_dit_synthetic_2.cpp
      tests/test_vulkan_dit_synthetic_3.cpp
      tests/test_vulkan_dit_integration.cpp
      tests/test_vulkan_attention_synthetic.cpp
      tests/test_vulkan_linear_synthetic.cpp
      tests/test_vulkan_runtime_synthetic.cpp
      tests/test_vulkan_codec_synthetic.cpp
      tests/test_vulkan_dit_checkpoint.cpp)
    target_link_libraries(slopfab_vulkan_tests PRIVATE slopfab_vulkan)
    set(SLOPFAB_AFFINE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp")
    set(SLOPFAB_AFFINE_SPV "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/affine.comp.spv")
    file(READ "${SLOPFAB_AFFINE_SOURCE}" SLOPFAB_AFFINE_SOURCE_TEXT)
    string(REPLACE "\r\n" "\n" SLOPFAB_AFFINE_SOURCE_TEXT "${SLOPFAB_AFFINE_SOURCE_TEXT}")
    string(SHA256 SLOPFAB_AFFINE_SOURCE_HASH "${SLOPFAB_AFFINE_SOURCE_TEXT}")
    file(SHA256 "${SLOPFAB_AFFINE_SPV}" SLOPFAB_AFFINE_SPV_HASH)
    if(NOT SLOPFAB_AFFINE_SOURCE_HASH STREQUAL
        "5ff2705b9c300adbdb476d393226f0dda0a3e23dd7b1eda9ac8095a9165345cd" OR
       NOT SLOPFAB_AFFINE_SPV_HASH STREQUAL
        "75c8343ac68d52af2c5908db040af9768f6abbce9df0a82268d131afbfc2ede5")
      message(FATAL_ERROR
        "slopfab: affine.comp or its checked-in SPIR-V changed. Regenerate with the "
        "documented glslang command, verify the shader test, and update both hashes")
    endif()
    target_compile_definitions(slopfab_vulkan_tests PRIVATE
      SLOPFAB_TEST_AFFINE_SPV_PATH="affine.comp.spv"
      SLOPFAB_TEST_ATTENTION_PREPARE_SPV_PATH="tensor_attention_prepare.comp.spv"
      SLOPFAB_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_custom_command(TARGET slopfab_vulkan_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLOPFAB_AFFINE_SPV}" "$<TARGET_FILE_DIR:slopfab_vulkan_tests>/affine.comp.spv"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLOPFAB_TENSOR_ATTENTION_PREPARE_SPV}"
        "$<TARGET_FILE_DIR:slopfab_vulkan_tests>/tensor_attention_prepare.comp.spv"
      VERBATIM)
    if(MSVC)
      target_compile_options(slopfab_vulkan_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
    else()
      target_compile_options(slopfab_vulkan_tests PRIVATE -Wall -Wextra)
    endif()
    slopfab_test_suite(vulkan slopfab_vulkan_tests checkpoint integration)
  endif()

  if(SLOPFAB_ENABLE_CUDA AND SLOPFAB_ENABLE_VULKAN)
    add_executable(slopfab_tensor_backend_tests
      tests/harness.cpp tests/allocation_guard.cpp
      tests/test_tensor_backends.cu
      tests/test_tensor_backends_reference_synthetic.cu
      tests/test_tensor_backends_arithmetic_synthetic.cu
      tests/test_tensor_backends_pointwise_synthetic_1.cu
      tests/test_tensor_backends_pointwise_synthetic_2.cu
      tests/test_tensor_backends_attention_synthetic_1.cu
      tests/test_tensor_backends_attention_synthetic_2.cu
      tests/test_tensor_backends_attention_benchmark.cu
      tests/test_tensor_backends_attention_checkpoint.cu
      tests/test_tensor_backends_dit_synthetic.cu
      tests/test_tensor_backends_conditioner_synthetic.cu
      tests/test_tensor_backends_conditioner_checkpoint.cu
      tests/test_tensor_backends_conditioner_integration_1.cu
      tests/test_tensor_backends_conditioner_integration_2.cu
      tests/test_tensor_backends_dit_checkpoint.cu
      tests/test_tensor_backends_dit_integration_1.cu
      tests/test_tensor_backends_dit_integration_2.cu
      tests/test_tensor_backends_pointwise_benchmark.cu
      tests/test_tensor_backends_normalization_synthetic.cu
      tests/test_tensor_backends_linear_synthetic.cu
      tests/test_tensor_backends_linear_checkpoint.cu
      tests/test_tensor_backends_linear_benchmark.cu
      tests/test_tensor_backends_vae_synthetic.cu
      tests/test_tensor_backends_reference_integration.cu
      tests/test_tensor_backends_vae_integration.cu tests/test_audio_vulkan.cu
      tests/test_audio_vulkan_checkpoint.cu
      tests/test_audio_vulkan_decoder_integration.cu
      tests/test_audio_vulkan_integration.cu tests/test_lora_backends.cu
      tests/test_vsa_vulkan.cu)
    target_link_libraries(slopfab_tensor_backend_tests PRIVATE
      slopfab_cuda slopfab_vulkan)
    target_compile_definitions(slopfab_tensor_backend_tests PRIVATE
      SLOPFAB_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(WIN32)
      target_link_libraries(slopfab_tensor_backend_tests PRIVATE bcrypt)
    endif()
    if(MSVC)
      target_compile_options(slopfab_tensor_backend_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8;/EHsc>)
    endif()
    slopfab_test_suite(tensor_backends slopfab_tensor_backend_tests checkpoint integration benchmark)
  endif()

  if(SLOPFAB_ENABLE_CUDA)
    add_executable(slopfab_cublas_dispatch_tests tests/test_cublas_dispatch.cpp)
    target_link_libraries(slopfab_cublas_dispatch_tests PRIVATE slopfab_cuda)
    if(WIN32)
      add_test(NAME cublas_cuda12 COMMAND slopfab_cublas_dispatch_tests 12)
      set_tests_properties(cublas_cuda12 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=12")
      add_test(NAME cublas_cuda13 COMMAND slopfab_cublas_dispatch_tests 13)
      set_tests_properties(cublas_cuda13 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=13")
      add_test(NAME cublas_auto_prefer13 COMMAND slopfab_cublas_dispatch_tests auto)
      set_tests_properties(cublas_auto_prefer13 PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=auto")
      add_test(NAME cublas_auto_load_fallback
        COMMAND slopfab_cublas_dispatch_tests auto-fallback)
      add_test(NAME cublas_explicit_load_failure
        COMMAND slopfab_cublas_dispatch_tests broken-13)
    else()
      add_executable(slopfab_cublas_linked_dispatch_tests
        tests/test_cublas_linked_dispatch.cpp)
      target_link_libraries(slopfab_cublas_linked_dispatch_tests PRIVATE slopfab_cuda)
      add_test(NAME cublas_linked_auto
        COMMAND slopfab_cublas_linked_dispatch_tests auto success)
      add_test(NAME cublas_linked_matching
        COMMAND slopfab_cublas_linked_dispatch_tests
          ${CUDAToolkit_VERSION_MAJOR} success)
      if(CUDAToolkit_VERSION_MAJOR EQUAL 12)
        set(SLOPFAB_CUBLAS_MISMATCH_MAJOR 13)
      else()
        set(SLOPFAB_CUBLAS_MISMATCH_MAJOR 12)
      endif()
      add_test(NAME cublas_linked_mismatch
        COMMAND slopfab_cublas_linked_dispatch_tests
          ${SLOPFAB_CUBLAS_MISMATCH_MAJOR} failure)
      add_test(NAME cublas_linked_environment_matching
        COMMAND slopfab_cublas_linked_dispatch_tests environment success)
      set_tests_properties(cublas_linked_environment_matching PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=${CUDAToolkit_VERSION_MAJOR}")
      add_test(NAME cublas_linked_environment_mismatch
        COMMAND slopfab_cublas_linked_dispatch_tests environment failure)
      set_tests_properties(cublas_linked_environment_mismatch PROPERTIES
        ENVIRONMENT "SLOPFAB_CUDA_VERSION=${SLOPFAB_CUBLAS_MISMATCH_MAJOR}")
    endif()

    add_executable(slopfab_kernel_tests
      tests/harness.cpp
      tests/test_kernels.cu
      tests/test_attention_plan.cu
      tests/test_nn_kernels.cu
      tests/test_nn_kernels_linear_synthetic.cu
      tests/test_nn_kernels_nvfp4_synthetic.cu
      tests/test_nn_kernels_attention_synthetic.cu
      tests/test_nn_kernels_attention_ranges_synthetic.cu
      tests/test_nn_kernels_workspace_synthetic.cu
      tests/test_nn_kernels_nvfp4_benchmark.cu
      tests/test_nn_kernels_operators_benchmark.cu
      tests/test_nn_kernels_vision_synthetic.cu
      tests/test_nn_kernels_vision_integration.cu
      tests/test_reference_encoder.cu
      tests/test_audio_vae.cu
        tests/test_transformer.cu
      tests/test_transformer_loading_synthetic.cu
      tests/test_transformer_denoise_synthetic.cu
      tests/test_transformer_checkpoint.cu
        tests/test_vsa.cu
      tests/test_encoder.cu
      tests/test_encoder_loading_synthetic.cu
      tests/test_encoder_checkpoint.cu
      tests/test_encoder_checkpoint_integration.cu
      tests/test_keyframe_encoder.cu
      tests/test_vit_decoder.cu
      tests/test_int8_weight.cu
    )
    target_link_libraries(slopfab_kernel_tests PRIVATE slopfab_cuda)
    slopfab_test_suite(kernels slopfab_kernel_tests checkpoint integration benchmark)

    if(WIN32)
      add_test(NAME cuda_version_option
        COMMAND slopfab --cuda-version=12 --help)
      add_test(NAME cuda_version_invalid
        COMMAND slopfab --cuda-version=invalid --help)
      set_tests_properties(cuda_version_invalid PROPERTIES WILL_FAIL TRUE)
    endif()
  endif()

  # The C ABI is tested through the DLL rather than by compiling capi.cpp into
  # the test binary, which is the only way the interesting failures show up:
  # a missing export, a mangled name, a calling-convention mismatch and a
  # struct the two sides lay out differently are all link- or run-time
  # properties of the module, invisible to a test that includes the source.
  if(SLOPFAB_BUILD_C_API)
    # Its own main, not tests/test_main.cpp: that file carries C++ test cases
    # of its own and would drag slopfab_core into a binary whose entire point is
    # linking nothing but the DLL.
    add_executable(slopfab_capi_tests tests/harness.cpp tests/test_capi.cpp)
    target_link_libraries(slopfab_capi_tests PRIVATE slopfab_c)
    target_include_directories(slopfab_capi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MSVC)
      target_compile_options(slopfab_capi_tests PRIVATE /W4 /permissive- /utf-8)
    else()
      target_compile_options(slopfab_capi_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME capi COMMAND slopfab_capi_tests)
    set_tests_properties(capi PROPERTIES LABELS integration SKIP_RETURN_CODE 77)

    add_executable(slopfab_capi_cuda_dispatch_tests tests/test_capi_cuda_dispatch.cpp)
    target_link_libraries(slopfab_capi_cuda_dispatch_tests PRIVATE slopfab_c)
    target_include_directories(slopfab_capi_cuda_dispatch_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(MSVC)
      target_compile_options(slopfab_capi_cuda_dispatch_tests PRIVATE
        /W4 /permissive- /utf-8)
    else()
      target_compile_options(slopfab_capi_cuda_dispatch_tests PRIVATE -Wall -Wextra)
    endif()
    if(WIN32)
      add_test(NAME capi_cublas_cuda12 COMMAND slopfab_capi_cuda_dispatch_tests 12)
      add_test(NAME capi_cublas_cuda13 COMMAND slopfab_capi_cuda_dispatch_tests 13)
    endif()
  endif()
endif()

