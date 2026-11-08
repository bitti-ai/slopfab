#include "../src/cli/update_package.h"
#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
  using namespace slopfab::cli::updater;
  namespace fs = std::filesystem;
  try {
    if (argc < 3)
      throw std::runtime_error("missing fixture arguments");
    const std::string operation = argv[1];
    if (operation == "signature" && argc == 5) {
      std::ifstream message_file(fs::u8path(argv[3]), std::ios::binary);
      std::ifstream signature_file(fs::u8path(argv[4]), std::ios::binary);
      const std::string message((std::istreambuf_iterator<char>(message_file)), {});
      const std::string signature((std::istreambuf_iterator<char>(signature_file)), {});
      verify_ed25519(argv[2], message, signature);
    } else if ((operation == "signed-release" || operation == "trusted-release") && argc == 5) {
      std::ifstream file(fs::u8path(argv[2]));
      std::ifstream signature_file(fs::u8path(argv[3]), std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(file)), {});
      const std::string signature((std::istreambuf_iterator<char>(signature_file)), {});
      const auto release = parse_release(slopfab::json::parse(text));
      if (operation == "trusted-release")
        verify_release_signature(release, signature);
      else
        verify_ed25519(argv[4], signature_message(release), signature);
    } else if (operation == "release" && argc == 4) {
      std::ifstream file(fs::u8path(argv[2]));
      const std::string text((std::istreambuf_iterator<char>(file)), {});
      auto release = parse_release(slopfab::json::parse(text));
      std::cout << (newer_version(release.version, argv[3]) ? "available" : "up_to_date");
    } else if (operation == "extract" && argc == 6) {
      verify_package(fs::u8path(argv[2]), argv[4], std::stoull(argv[5]));
      extract_package(fs::u8path(argv[2]), fs::u8path(argv[3]));
    } else if (operation == "install" && argc == 5) {
      install_package(fs::u8path(argv[2]), fs::u8path(argv[3]), fs::u8path(argv[4]));
    } else
      throw std::runtime_error("invalid fixture arguments");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
