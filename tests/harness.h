// Minimal test harness, shared by every test translation unit.
//
// Hand-rolled rather than gtest to keep the dependency count at zero. The one
// thing it adds over a plain `main` calling functions in order is *self
// registration*: each test file contributes its cases from its own translation
// unit, so several people can add tests without ever editing a shared file.
//
// Use it as:
//
//   VIDFAB_TEST(my_thing) {
//     CHECK(1 + 1 == 2);
//     CHECK_NEAR(x, 3.0, 1e-6);
//     CHECK_CLOSE(expected_vec, actual_vec, 1e-5, "my kernel");
//   }
//
// Failures print `file:line` plus the two offending values, and — for vector
// comparisons — the index of the worst element. That last detail matters more
// than it looks: the failures this project cares about are silently wrong
// layouts, and "worst error at index 8192 of 16384" localises a swapped half
// far faster than a scalar max-error does.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vidfab::test {

using TestFn = void (*)();

// Registers a case. Returns true so it can be used in a namespace-scope
// initialiser. Registration order across translation units is unspecified;
// tests must not depend on each other.
bool register_test(const char* name, TestFn fn);

void set_current(const char* name);

void check(bool ok, const char* expr, const char* file, int line);
void check_near(double a, double b, double tol, const char* expr, const char* file, int line);

// Elementwise comparison with a max-absolute-error tolerance. Reports the
// worst index, which is what localises a layout bug.
void check_close(const std::vector<float>& expected, const std::vector<float>& actual, double tol,
                 const char* what, const char* file, int line);

// As check_close, but passes if *either* the absolute or the relative error is
// within tolerance elementwise. Use for quantised paths, where small values
// carry large relative error and large values carry large absolute error.
void check_close_rel(const std::vector<float>& expected, const std::vector<float>& actual,
                     double abs_tol, double rel_tol, const char* what, const char* file, int line);

// A check for a known, deliberately deferred defect.
//
// It runs, it reports the real measured number every time, and it does NOT
// fail the suite. Use it only with a written justification at the call site
// naming the old threshold, the observed value and the reason for deferring —
// the point is that the next reader sees a decision, not a loose tolerance.
//
// This exists because the two bad options are worse. Widening a threshold
// silently turns a real signal into a permanently green lie; deleting the
// assertion loses the detector that found the problem. A deferred check keeps
// measuring and keeps complaining, but lets an unrelated change ship.
void check_deferred(bool ok, const char* file, int line, const char* fmt, ...);

// Counts one check and, when it fails, prints a caller-formatted explanation.
// For assertions whose useful diagnostic is not "expected vs actual" — a row
// that fails to sum to one, a count of mismatching bit patterns.
void check_printf(bool ok, const char* file, int line, const char* fmt, ...);

// True when `fn` threw anything derived from std::exception.
bool throws(TestFn fn);

// Deterministic pseudo-random fill. Avoids <random>, whose engines are
// portable but whose distributions are not, so a golden value computed on one
// standard library reproduces on another.
std::vector<float> make_data(size_t n, uint32_t seed, float scale = 1.0f);

// Runs everything registered. Returns a process exit code.
int run_all();

int check_count();
int failure_count();
int deferred_count();

}  // namespace vidfab::test

#define CHECK(expr) ::vidfab::test::check((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
  ::vidfab::test::check_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)
#define CHECK_CLOSE(e, a, tol, what) \
  ::vidfab::test::check_close((e), (a), (tol), (what), __FILE__, __LINE__)
#define CHECK_CLOSE_REL(e, a, atol, rtol, what) \
  ::vidfab::test::check_close_rel((e), (a), (atol), (rtol), (what), __FILE__, __LINE__)
#define CHECK_MSG(ok, ...) ::vidfab::test::check_printf((ok), __FILE__, __LINE__, __VA_ARGS__)
// Known-failing on purpose. Reports the number, never fails the run.
#define CHECK_DEFERRED(ok, ...) \
  ::vidfab::test::check_deferred((ok), __FILE__, __LINE__, __VA_ARGS__)

// Names the case currently running, for files that register their functions
// separately rather than through VIDFAB_TEST.
#define TEST(name) ::vidfab::test::set_current(name)

// Defines and registers a test case in one go.
#define VIDFAB_TEST(name)                                                              \
  static void vidfab_test_##name();                                                    \
  static const bool vidfab_test_##name##_registered =                                  \
      ::vidfab::test::register_test(#name, &vidfab_test_##name);                       \
  static void vidfab_test_##name()
