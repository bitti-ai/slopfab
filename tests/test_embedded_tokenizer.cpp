// This executable intentionally has no tokenizer resource supplied by its
// caller. On Linux the data must survive static archive extraction into the
// consuming executable, just as it must when linked into libslopfab.so.
#include "slopfab/text/tokenizer.h"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv) {
  try {
    slopfab::text::Tokenizer embedded;
    if (argc == 1) {
      try {
        embedded.load_embedded();
      } catch (const std::exception& e) {
        const std::string error = e.what();
        if (error.find("--tokenizer") != std::string::npos &&
            error.find("SLOPFAB_EMBED_TOKENIZER") != std::string::npos)
          return 0;
        throw;
      }
      std::fprintf(stderr, "embedding disabled but load_embedded succeeded\n");
      return 1;
    }

    // Load before opening any file: embedded loading must not depend on the
    // current directory or a runtime tokenizer.json lookup.
    embedded.load_embedded();
    slopfab::text::Tokenizer file;
    file.load(argv[1]);
    if (!embedded.loaded() || embedded.source_sha256().size() != 64 ||
        embedded.source_sha256() != file.source_sha256() ||
        embedded.vocab_size() != file.vocab_size() ||
        embedded.merge_ranks_for_testing() != file.merge_ranks_for_testing()) {
      std::fprintf(stderr, "embedded tokenizer vocabulary or merges differ\n");
      return 1;
    }
    for (size_t i = 0; i < file.vocab_size(); ++i) {
      const auto id = static_cast<int32_t>(i);
      if (embedded.id_to_token(id) != file.id_to_token(id) ||
          embedded.token_to_id(file.id_to_token(id)) != file.token_to_id(file.id_to_token(id))) {
        std::fprintf(stderr, "embedded tokenizer differs at token %zu\n", i);
        return 1;
      }
    }
    for (const char* prompt : {"hello world", "<|im_start|>system<|im_end|>", "line1\r\nline2\t123",
                               "\xe4\xb8\xad\xe6\x96\x87"}) {
      const auto ids = embedded.encode(prompt);
      if (ids != file.encode(prompt) || embedded.decode(ids) != file.decode(ids)) {
        std::fprintf(stderr, "embedded tokenizer encoding differs\n");
        return 1;
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
