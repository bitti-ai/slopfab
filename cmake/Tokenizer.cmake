# The tokenizer is a build input, never a runtime download. The same option
# controls the CLI and the shared C API so an empty path behaves consistently.
option(SLOPFAB_EMBED_TOKENIZER "Embed a tokenizer in the CLI and C API" ON)
set(SLOPFAB_TOKENIZER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ref/text_encoder/tokenizer.json"
    CACHE FILEPATH "Tokenizer JSON to embed when SLOPFAB_EMBED_TOKENIZER is enabled")

set(SLOPFAB_TOKENIZER_RESOURCES)
target_compile_definitions(slopfab_core PRIVATE
  SLOPFAB_EMBED_TOKENIZER=$<BOOL:${SLOPFAB_EMBED_TOKENIZER}>)

if(SLOPFAB_EMBED_TOKENIZER)
  get_filename_component(SLOPFAB_TOKENIZER_RESOURCE "${SLOPFAB_TOKENIZER_FILE}"
    ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${SLOPFAB_TOKENIZER_RESOURCE}" OR
      IS_DIRECTORY "${SLOPFAB_TOKENIZER_RESOURCE}")
    message(FATAL_ERROR
      "slopfab: tokenizer JSON is missing: ${SLOPFAB_TOKENIZER_RESOURCE}. "
      "Set -DSLOPFAB_TOKENIZER_FILE=/path/to/tokenizer.json, or configure "
      "-DSLOPFAB_EMBED_TOKENIZER=OFF and supply --tokenizer at runtime.")
  endif()
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
  if(WIN32)
    enable_language(RC)
    file(TO_CMAKE_PATH "${SLOPFAB_TOKENIZER_RESOURCE}" SLOPFAB_TOKENIZER_RESOURCE)
    configure_file(src/slopfab_tokenizer.rc.in generated/slopfab_tokenizer.rc @ONLY)
    # Resources must be attached to each final module, not a static archive.
    set(SLOPFAB_TOKENIZER_RESOURCES
      "${CMAKE_CURRENT_BINARY_DIR}/generated/slopfab_tokenizer.rc")
    set_property(SOURCE "${SLOPFAB_TOKENIZER_RESOURCES}" APPEND PROPERTY
      OBJECT_DEPENDS "${SLOPFAB_TOKENIZER_RESOURCE}")
  else()
    set(SLOPFAB_TOKENIZER_CPP "${CMAKE_CURRENT_BINARY_DIR}/generated/tokenizer_data.cpp")
    add_custom_command(OUTPUT "${SLOPFAB_TOKENIZER_CPP}"
      COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${SLOPFAB_TOKENIZER_RESOURCE}" "-DOUTPUT=${SLOPFAB_TOKENIZER_CPP}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedTokenizer.cmake"
      DEPENDS "${SLOPFAB_TOKENIZER_RESOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedTokenizer.cmake"
      COMMENT "Embedding tokenizer JSON" VERBATIM)
    target_sources(slopfab_core PRIVATE "${SLOPFAB_TOKENIZER_CPP}")
  endif()
endif()

function(slopfab_add_tokenizer_tests)
  add_executable(slopfab_embedded_tokenizer_tests tests/test_embedded_tokenizer.cpp
    ${SLOPFAB_TOKENIZER_RESOURCES})
  target_link_libraries(slopfab_embedded_tokenizer_tests PRIVATE slopfab_core)
  if(SLOPFAB_EMBED_TOKENIZER)
    add_test(NAME embedded_tokenizer COMMAND slopfab_embedded_tokenizer_tests
      "${SLOPFAB_TOKENIZER_RESOURCE}")
  else()
    add_test(NAME embedded_tokenizer_disabled COMMAND slopfab_embedded_tokenizer_tests)
  endif()
endfunction()
