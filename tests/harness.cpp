#include "harness.h"

#include <cstdarg>
#include <cstdio>
#include <exception>
#include <vector>

namespace vidfab::test {
namespace {

struct Case {
  const char* name;
  TestFn fn;
};

// Function-local static: registration happens during dynamic initialisation of
// other translation units, so the container must be constructed on first use
// rather than at namespace scope, where the order would be unspecified.
std::vector<Case>& cases() {
  static std::vector<Case> v;
  return v;
}

int g_checks = 0;
int g_failures = 0;
int g_deferred = 0;
int g_skipped = 0;
int g_skipped_fixture = 0;
int g_skipped_vram = 0;
const char* g_current = "";

// Strips the directory so failures read `test_kernels.cu:412` rather than an
// absolute path that differs between the build machine and a worktree.
const char* basename(const char* path) {
  const char* out = path;
  for (const char* p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\') out = p + 1;
  }
  return out;
}

}  // namespace

bool register_test(const char* name, TestFn fn) {
  cases().push_back({name, fn});
  return true;
}

void set_current(const char* name) { g_current = name; }

void check(bool ok, const char* expr, const char* file, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL [%s] %s:%d  %s\n", g_current, basename(file), line, expr);
  }
}

void check_near(double a, double b, double tol, const char* expr, const char* file, int line) {
  ++g_checks;
  if (!(std::fabs(a - b) <= tol)) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL [%s] %s:%d  %s  (%.9g vs %.9g, tol %g)\n", g_current,
                 basename(file), line, expr, a, b, tol);
  }
}

void check_close(const std::vector<float>& expected, const std::vector<float>& actual, double tol,
                 const char* what, const char* file, int line) {
  ++g_checks;
  if (expected.size() != actual.size()) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL [%s] %s:%d  %s: size %zu vs %zu\n", g_current, basename(file),
                 line, what, expected.size(), actual.size());
    return;
  }
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double d = std::fabs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    if (d > worst) {
      worst = d;
      worst_i = i;
    }
  }
  if (!(worst <= tol)) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL [%s] %s:%d  %s: max abs err %.3e at %zu of %zu (%.6g vs %.6g)\n",
                 g_current, basename(file), line, what, worst, worst_i, expected.size(),
                 expected[worst_i], actual[worst_i]);
  }
}

void check_close_rel(const std::vector<float>& expected, const std::vector<float>& actual,
                     double abs_tol, double rel_tol, const char* what, const char* file,
                     int line) {
  ++g_checks;
  if (expected.size() != actual.size()) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL [%s] %s:%d  %s: size %zu vs %zu\n", g_current, basename(file),
                 line, what, expected.size(), actual.size());
    return;
  }
  double worst_score = 0.0;  // err / allowance; > 1 fails
  size_t worst_i = 0;
  double worst_abs = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double e = static_cast<double>(expected[i]);
    const double a = static_cast<double>(actual[i]);
    const double d = std::fabs(e - a);
    const double allow = abs_tol + rel_tol * std::fabs(e);
    const double score = allow > 0.0 ? d / allow : (d > 0.0 ? 1e30 : 0.0);
    if (score > worst_score) {
      worst_score = score;
      worst_i = i;
      worst_abs = d;
    }
  }
  if (worst_score > 1.0) {
    ++g_failures;
    std::fprintf(stderr,
                 "  FAIL [%s] %s:%d  %s: worst at %zu of %zu, abs err %.3e (%.6g vs %.6g), "
                 "%.2fx allowance\n",
                 g_current, basename(file), line, what, worst_i, expected.size(), worst_abs,
                 expected[worst_i], actual[worst_i], worst_score);
  }
}

void check_deferred(bool ok, const char* file, int line, const char* fmt, ...) {
  ++g_checks;
  if (ok) return;
  ++g_deferred;
  std::fprintf(stderr, "  DEFER [%s] %s:%d  ", g_current, basename(file), line);
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fputc(0x0A, stderr);
}

void skip(SkipReason reason, const char* file, int line, const char* fmt, ...) {
  ++g_skipped;
  if (reason == SkipReason::kMissingFixture) {
    ++g_skipped_fixture;
  } else {
    ++g_skipped_vram;
  }
  std::fprintf(stderr, "  SKIP [%s] %s:%d  (%s) ", g_current, basename(file), line,
               reason == SkipReason::kMissingFixture ? "fixture absent" : "insufficient vram");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fputc(0x0A, stderr);
}

int skipped_count() { return g_skipped; }

void check_printf(bool ok, const char* file, int line, const char* fmt, ...) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  std::fprintf(stderr, "  FAIL [%s] %s:%d  ", g_current, basename(file), line);
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fputc('\n', stderr);
}

bool throws(TestFn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::vector<float> make_data(size_t n, uint32_t seed, float scale) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v[i] = ((static_cast<float>(s & 0xFFFFFFu) / 16777216.0f) - 0.5f) * 2.0f * scale;
  }
  return v;
}

int check_count() { return g_checks; }
int failure_count() { return g_failures; }
int deferred_count() { return g_deferred; }

int run_all() {
  const char* filter = std::getenv("VIDFAB_TEST_FILTER");
  for (const Case& c : cases()) {
    if (filter != nullptr && std::strstr(c.name, filter) == nullptr) continue;
    g_current = c.name;
    std::printf("test %s\n", c.name);
    try {
      c.fn();
    } catch (const std::exception& e) {
      ++g_failures;
      std::fprintf(stderr, "  FAIL [%s] threw: %s\n", c.name, e.what());
    }
  }
  // Deferred checks get their own place in the summary. Folding them into
  // "failures" would block unrelated work; folding them into "passes" would
  // make a known defect invisible. They are neither.
  std::printf("\n%d checks, %d failures", g_checks, g_failures);
  if (g_deferred != 0) {
    std::printf(", %d DEFERRED (known defects, see DEFER lines above)", g_deferred);
  }
  // A skipped case contributes no checks, so without this the run reports
  // success and the missing coverage is invisible. The split matters: a
  // fixture skip is fixed by fetching a file, a vram skip by freeing the card,
  // and only the second makes an otherwise-identical run report fewer checks.
  if (g_skipped != 0) {
    std::printf(", %d skipped (%d fixture, %d vram; see SKIP lines above)", g_skipped,
                g_skipped_fixture, g_skipped_vram);
  }
  std::fputc(0x0A, stdout);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace vidfab::test
