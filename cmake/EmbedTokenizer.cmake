# Encode bytes rather than inserting JSON into a raw literal: this preserves
# CRLF, Unicode and delimiter-like contents exactly, regardless of the source
# encoding or compiler locale. Generated at build time using CMake alone.
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "EmbedTokenizer.cmake requires INPUT and OUTPUT")
endif()
file(SIZE "${INPUT}" tokenizer_size)
if(tokenizer_size EQUAL 0)
  message(FATAL_ERROR "slopfab: tokenizer JSON is empty: ${INPUT}")
endif()
set(tokenizer_temp "${OUTPUT}.tmp")
file(WRITE "${tokenizer_temp}" "// Generated tokenizer bytes. Do not edit.\n#include <string_view>\nnamespace slopfab::text::detail {\nnamespace {\nconst char tokenizer_data[] =\n")
set(tokenizer_offset 0)
while(tokenizer_offset LESS tokenizer_size)
  # Bound the generator's working set even for larger tokenizer vocabularies.
  file(READ "${INPUT}" tokenizer_hex OFFSET ${tokenizer_offset} LIMIT 16384 HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "\\\\x\\1" tokenizer_escaped "${tokenizer_hex}")
  file(APPEND "${tokenizer_temp}" "\"${tokenizer_escaped}\"\n")
  math(EXPR tokenizer_offset "${tokenizer_offset} + 16384")
endwhile()
file(APPEND "${tokenizer_temp}" ";\n}\nstd::string_view embedded_tokenizer_json() {\n  return {tokenizer_data, sizeof(tokenizer_data) - 1};\n}\n}\n")
file(RENAME "${tokenizer_temp}" "${OUTPUT}")
