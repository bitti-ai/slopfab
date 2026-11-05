// Local SSH stand-in: execute the exact remote shell command with inherited
// stdin/stdout. Optional canned GPU events exercise downloads without weights.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <sys/wait.h>
#endif

int main(int argc, char** argv) {
  if (argc != 12 || std::string(argv[1]) != "-T" || std::string(argv[3]) != "BatchMode=yes" ||
      std::string(argv[10]) != "fixture")
    return 90;
  if (std::getenv("SLOPFAB_TEST_SSH_FAIL"))
    return 255;
  std::string command = argv[11], decoded = command;
  const auto encoded = command.find("-EncodedCommand ");
  if (encoded != std::string::npos) {
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    unsigned bits = 0;
    int count = 0;
    std::string bytes;
    for (char c : command.substr(encoded + 16)) {
      if (c == '=')
        break;
      auto index = alphabet.find(c);
      if (index == std::string::npos)
        return 91;
      bits = (bits << 6) | unsigned(index);
      count += 6;
      if (count >= 8) {
        count -= 8;
        bytes += char((bits >> count) & 255);
      }
    }
    decoded.clear();
    for (size_t i = 0; i < bytes.size(); i += 2)
      decoded += bytes[i];
  }
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  if (decoded.find("_fetch") != std::string::npos && std::getenv("SLOPFAB_TEST_TRUNCATE")) {
    std::fwrite("broken", 1, 6, stdout);
    return 0;
  }
  if (decoded.find("_run") != std::string::npos) {
    if (const auto* reply = std::getenv("SLOPFAB_TEST_REPLY")) {
      std::ifstream input(reply, std::ios::binary);
      char bytes[4096];
      while (input.read(bytes, sizeof(bytes)) || input.gcount())
        std::fwrite(bytes, 1, size_t(input.gcount()), stdout);
      return input.bad() ? 92 : 0;
    }
  }
  const auto status = std::system(command.c_str());
#ifdef _WIN32
  return status;
#else
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}
