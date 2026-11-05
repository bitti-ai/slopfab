#include "cli/commands.h"
#include "cli/agent.h"
#include "cli/agent_instructions.h"
#include "cli/settings.h"
#include "cli/mods.h"
#include "cli/script.h"
#include "cli/motions.h"
#include "cli/frames.h"
#include "cli/projects.h"
#include "cli/prompting.h"
#include "cli/transcribe.h"
#include "cli/remote.h"
#include <cstdio>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
using namespace slopfab::cli;

int cli_main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "remote" &&
      (std::string(argv[2]) == "_run" || std::string(argv[2]) == "_fetch"))
    return remote_worker(argv[0], std::string(argv[2]) == "_fetch");
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  bool machine = false;
  std::vector<std::string> args;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--") {
        for (; i < argc; ++i) args.push_back(argv[i]);
        break;
      }
      if (a == "--agent") {
        machine = true;
        continue;
      }
      if (a == "--config" || a == "--jobs-dir") {
        if (++i == argc)
          throw CliError("INVALID_REQUEST", "missing value for " + a);
        auto path = std::filesystem::absolute(std::filesystem::u8path(argv[i])).u8string();
        if (a == "--config")
          agent().config_path = path;
        else
          agent().jobs_path = path;
        continue;
      }
      args.push_back(a);
      // Keep option values opaque: a prompt equal to "--agent" is still a prompt.
      if (option_takes_value(a,
                             args.empty() ? std::string_view{} : std::string_view(args.front())) &&
          i + 1 < argc)
        args.push_back(argv[++i]);
    }
    initialize_agent(machine,
                     std::filesystem::absolute(std::filesystem::u8path(argv[0])).u8string());
    ensure_agent_instructions();
    ensure_settings();
    if (args.empty()) {
      print_usage();
      return finish_command(0);
    }
#if SLOPFAB_WITH_CUDA
    std::vector<char*> cuda_args{argv[0]};
    for (auto& a : args)
      cuda_args.push_back(a.data());
    cuda_args.push_back(nullptr);
    int cuda_argc = int(cuda_args.size() - 1);
    consume_cuda_version_option(cuda_argc, cuda_args.data());
    std::vector<std::string> filtered;
    for (int i = 1; i < cuda_argc; ++i)
      filtered.emplace_back(cuda_args[i]);
    args = std::move(filtered);
    if (args.empty()) {
      print_usage();
      return finish_command(0);
    }
#endif
    const std::string command = args.front();
    args.erase(args.begin());
    std::vector<char*> av;
    for (auto& a : args)
      av.push_back(a.data());
    const int ac = int(av.size());
    if (command == "help" || command == "--help" || command == "-h") {
      if (args.empty()) {
        print_usage(true, mod_commands());
        return finish_command(0);
      }
      if (const auto* help = find_command(args[0]))
        return print_command_help(*help);
      if (args.size() == 1 && try_mod_command(args[0], {}, true))
        return 0;
      throw CliError("INVALID_REQUEST", "unknown command: " + args[0]);
    }
    if (const auto* help = find_command(command); help && wants_help(ac, av.data(), command))
      return print_command_help(*help);
    if (command == "mods")
      return mods_command(args);
    if (command == "run")
      return script_command(args);
    if (command == "motions")
      return motions_command(args);
    if (command == "video") {
      if (args.empty() || args.front() != "frames")
        throw CliError("INVALID_REQUEST", "use slopfab video frames; see slopfab help video");
      args.erase(args.begin());
      return frames_command(args);
    }
    if (command == "prompts")
      return prompts_command(args);
    if (command == "transcribe")
      return transcribe_command(args);
    if (command == "projects")
      return projects_command(args);
    if (command == "remote")
      return remote_command(args);
    if (command == "settings")
      return settings_command(args);
    if (command == "generate" || command == "validate") {
      if (command == "validate")
        args.push_back("--dry-run");
      return finish_command(run_generation_arguments(args));
    }
    if (command == "capabilities" || command == "schema" || command == "models" ||
        command == "doctor" || command == "devices")
      return discovery_command(command, args);
    if (command == "submit" || command == "status" || command == "wait" || command == "cancel" ||
        command == "worker")
      return job_command(command, args);
    if (command == "version") {
      if (!args.empty())
        throw CliError("INVALID_REQUEST", "version takes no arguments");
      emit({{"type", str("result")}, {"ok", Value(true)}, {"version", str(kVersion)}}, true);
      return 0;
    }
    if (try_mod_command(command, args))
      return 0;
    // Older diagnostic tools return their text inside the JSON envelope. Their
    // internal output never leaks onto the machine protocol stream.
    FILE* capture = nullptr;
    int saved = -1;
    if (machine) {
      capture = std::tmpfile();
      if (!capture)
        throw std::runtime_error("cannot capture command result");
      std::fflush(stdout);
#ifdef _WIN32
      saved = _dup(_fileno(stdout));
      _dup2(_fileno(capture), _fileno(stdout));
#else
      saved = dup(fileno(stdout));
      dup2(fileno(capture), fileno(stdout));
#endif
    }

    struct CaptureRestore {
      FILE* file;
      int fd;

      ~CaptureRestore() {
        if (file) {
          std::fflush(stdout);
#ifdef _WIN32
          _dup2(fd, _fileno(stdout));
          _close(fd);
#else
          dup2(fd, fileno(stdout));
          close(fd);
#endif
          std::fclose(file);
        }
      }
    } restore{capture, saved};

    int status = 0;
    if (command == "upscale")
      status = cmd_upscale(ac, av.data(), argv[0]);
    else if (command == "upscale-latents")
      status = cmd_upscale_latents(ac, av.data());
    else if (command == "prepare-lora")
      status = cmd_prepare_lora(ac, av.data());
    else if (command == "inspect")
      status = cmd_inspect(ac, av.data());
    else if (command == "compare")
      status = cmd_compare(ac, av.data());
    else if (command == "compare-y4m")
      status = cmd_compare_y4m(ac, av.data());
    else if (command == "tokenize")
      status = cmd_tokenize(ac, av.data());
    else if (command == "encode-text")
      status = cmd_encode_text(ac, av.data(), argv[0]);
    else if (command == "encode-refmod")
      status = cmd_encode_refmod(ac, av.data(), argv[0]);
    else if (command == "bundle-refmods")
      status = cmd_bundle_refmods(ac, av.data());
#if SLOPFAB_WITH_CUDA || SLOPFAB_WITH_VULKAN
    else if (command == "decode")
      status = cmd_decode(ac, av.data());
#endif
    else
      throw CliError("INVALID_REQUEST", "unknown command: " + command + "; run slopfab help");
    if (capture && status == 0) {
      std::fflush(stdout);
      std::rewind(capture);
      std::string text;
      char bytes[4096];
      size_t n;
      while ((n = std::fread(bytes, 1, sizeof(bytes), capture)))
        text.append(bytes, n);
      emit({{"type", str("result")}, {"ok", Value(true)}, {"text", str(text)}}, true);
    }
    return finish_command(status);
  } catch (const std::exception& e) {
    if (machine && !agent().enabled)
      initialize_agent(true, argv[0]);
    return report_exception(e);
  }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** wide) {
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) {
    int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, nullptr, 0, nullptr,
                                    nullptr);
    if (bytes <= 0)
      return 2;
    std::string value(size_t(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, value.data(), bytes, nullptr,
                        nullptr);
    value.pop_back();
    args.push_back(std::move(value));
  }
  std::wstring module(32768, L'\0');
  auto n = GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
  if (n > 0 && n < module.size()) {
    module.resize(n);
    args[0] = std::filesystem::path(module).u8string();
  }
  std::vector<char*> av;
  for (auto& a : args)
    av.push_back(a.data());
  return cli_main(argc, av.data());
}
#else
int main(int argc, char** argv) {
  std::error_code ec;
  auto module = std::filesystem::read_symlink("/proc/self/exe", ec);
  std::string executable;
  if (!ec) {
    executable = module.u8string();
    argv[0] = executable.data();
  }
  return cli_main(argc, argv);
}
#endif
