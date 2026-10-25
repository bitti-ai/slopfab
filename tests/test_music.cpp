#include "harness.h"
#include "slopfab/music.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "../src/text/nfc.h"
#include <limits>

SLOPFAB_TEST(music_plan_and_guards) {
  using namespace slopfab;
  GenerateRequest r;
  r.model_family = "yue2";
  r.music.duration = 1.01;
  auto p = resolve_plan(r);
  CHECK(p.model.family == "yue2");
  CHECK(p.layout.num_audio_rows == 26);
  CHECK(p.layout.num_video_rows == 0);
  CHECK(p.num_inference_steps == 32);
  CHECK(p.geometry.audio_sample_rate == 48000);
  CHECK(p.geometry.audio_channels == 2);
  auto fails = [](const auto& fn) {
    try {
      fn();
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  r.reference_image_paths = {"image.png"};
  CHECK(fails([&] {
    resolve_plan(r);
  }));
  r.reference_image_paths.clear();
  r.music.duration = std::numeric_limits<double>::quiet_NaN();
  CHECK(fails([&] {
    resolve_plan(r);
  }));
  r.music.duration = 301;
  CHECK(fails([&] {
    resolve_plan(r);
  }));
  r.music.duration = 30;
  r.music.planning = "off";
  r.music.abc = "X:1";
  CHECK(fails([&] {
    resolve_plan(r);
  }));
  r.music.abc.clear();
  RunOptions o;
  o.inference_backend = DeviceBackend::kVulkan;
  CHECK(fails([&] {
    validate_generation_options(r, p, o);
  }));
  CHECK(fails([] {
    parse_music_settings(R"({"top_k":1.5})");
  }));
  CHECK(fails([] {
    parse_music_settings(R"({"unknown":true})");
  }));
  auto s = parse_music_settings(
      R"({"lyrics":"[verse]\nHello", "planning":"off", "duration":8,"temperature":0})");
  CHECK(s.lyrics == "[verse]\nHello");
  CHECK(s.duration == 8);
  CHECK(s.temperature == 0);
}

SLOPFAB_TEST(music_unicode_nfc) {
  using slopfab::text::normalize_nfc;
  CHECK(normalize_nfc("Cafe\xcc\x81") == "Caf\xc3\xa9");
  CHECK(normalize_nfc("\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8") == "\xea\xb0\x81");
  CHECK(normalize_nfc("a\xcc\x81\xcc\xa7") == "\xc3\xa1\xcc\xa7");
  CHECK(normalize_nfc("\xe2\x84\xab") == "\xc3\x85");
  CHECK(normalize_nfc("\xcc\x81\xcc\xa7") == "\xcc\xa7\xcc\x81");
}
