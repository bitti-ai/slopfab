// The second-order sampler.
//
// `FlowScheduler::step` is Euler by default and Adams-Bashforth 2 on request.
// AB2 costs exactly one model evaluation per step, like Euler, so the only
// reason to have it is that it should reach the same accuracy on fewer steps.
// Whether it does on a *distilled* checkpoint is an empirical question these
// tests cannot answer; what they can do is prove the integrator is the one it
// claims to be, and that the default path did not move.
//
// Three properties are worth stating separately because each fails silently:
//
//   1. On a constant velocity field the extrapolation 1.5*v - 0.5*v_prev is
//      identically v, so AB2 must reproduce Euler *bitwise*. Any coefficient
//      pair that does not sum to one, and any sign error on the history term,
//      breaks this.
//   2. On a linear ODE with a state-dependent velocity the extrapolation is
//      genuinely exercised, and AB2 must converge faster than Euler. This is
//      what separates "reduces to Euler" from "is second order".
//   3. Euler must be untouched. The test transcribes the pre-change update and
//      demands bit equality over whole trajectories at both live shifts.

#include <cmath>
#include <cstdio>
#include <vector>

#include "harness.h"
#include "vidfab/generate.h"
#include "vidfab/sampler/scheduler.h"

using vidfab::sampler::FlowScheduler;
using vidfab::sampler::SamplerKind;

namespace {

// The update as it stood before the sampler existed, transcribed from
// scheduler.cpp at 2480c82 rather than refactored out of it — a copy is the
// point, since a shared helper would move with the code it is meant to pin.
std::vector<float> reference_euler_step(const FlowScheduler& s, int i,
                                        const std::vector<float>& x,
                                        const std::vector<float>& v) {
  const float sigma_from_timestep = 1.0f - s.timesteps()[static_cast<size_t>(i)];
  const float ratio =
      s.sigmas()[static_cast<size_t>(i) + 1] / s.sigmas()[static_cast<size_t>(i)];
  std::vector<float> out(x.size());
  for (size_t j = 0; j < x.size(); ++j) {
    const float denoised = x[j] + sigma_from_timestep * v[j];
    out[j] = ratio * x[j] + (1.0f - ratio) * denoised;
  }
  return out;
}

// A velocity field that depends on the state, so an error anywhere in the
// trajectory propagates rather than cancelling: v = -k*x.
//
// In this scheduler's convention the effective update is
// x_{n+1} = x_n + h_n*v with h_n = sigma_n - sigma_{n+1}, so v = -k*x
// integrates dx/dsigma = k*x. Starting from x = 1 at sigma = 1 and running the
// grid down to its terminal sigma = 0, the exact answer is exp(-k) — a genuine
// linear ODE with a closed form, which is what makes a convergence order
// measurable rather than merely plausible.
double linear_ode(float shift, int grid_points, SamplerKind kind, float k) {
  FlowScheduler s(shift);
  s.set_timesteps(grid_points);
  s.set_sampler(kind);
  float x = 1.0f;
  const int steps = static_cast<int>(s.num_steps());
  for (int i = 0; i < steps; ++i) {
    const float v = -k * x;
    s.step(i, &x, &v, 1, &x);  // out aliases sample, which step permits
  }
  return static_cast<double>(x);
}

// The same integration with a deliberately wrong extrapolation, so the test can
// assert the implementation does *not* match a plausible mistake.
double linear_ode_wrong(float shift, int grid_points, float k, float c_now, float c_prev) {
  FlowScheduler s(shift);
  s.set_timesteps(grid_points);
  float x = 1.0f;
  float prev_v = 0.0f;
  bool have_prev = false;
  const int steps = static_cast<int>(s.num_steps());
  for (int i = 0; i < steps; ++i) {
    const float v = -k * x;
    const float v_hat = have_prev ? c_now * v + c_prev * prev_v : v;
    prev_v = v;
    have_prev = true;
    s.step(i, &x, &v_hat, 1, &x);  // euler mode: whatever velocity it is handed
  }
  return static_cast<double>(x);
}

}  // namespace

// The campaign lead's required test, in its exact form: AB2 with a first-order
// first step reproduces Euler on a linear ODE with a constant velocity field.
// With v_n == v_{n-1} the extrapolation 1.5*v - 0.5*v collapses to v *exactly*
// in fp32 (both products are exact scalings by powers of two plus a half, and
// the difference of the two is the original value), so this is a bit equality
// and not a tolerance. It catches a sign error on either term and any
// coefficient pair that does not sum to one.
VIDFAB_TEST(ab2_reproduces_euler_on_a_constant_velocity_field) {
  for (const float shift : {12.0f, 3.0f}) {
    for (const int grid : {6, 20, 50}) {
      FlowScheduler euler(shift), ab2(shift);
      euler.set_timesteps(grid);
      ab2.set_timesteps(grid);
      ab2.set_sampler(SamplerKind::kAb2);

      const std::vector<float> v = {0.75f, -0.4f, 3.25f, 0.0f, -1e-3f};
      std::vector<float> xe = {1.0f, -2.0f, 0.5f, 7.5f, -0.125f};
      std::vector<float> xa = xe;

      const int steps = static_cast<int>(euler.num_steps());
      for (int i = 0; i < steps; ++i) {
        euler.step(i, xe.data(), v.data(), xe.size(), xe.data());
        ab2.step(i, xa.data(), v.data(), xa.size(), xa.data());
      }
      CHECK_CLOSE(xe, xa, 0.0, "ab2 on a constant velocity field is bitwise Euler");
    }
  }
}

// The first step of a trajectory has no history, so it must be plain Euler —
// not AB2 with an invented v_{-1} of zero, which would scale the first step by
// 1.5 and is the obvious way to get this wrong.
VIDFAB_TEST(ab2_first_step_is_first_order) {
  FlowScheduler euler(12.0f), ab2(12.0f);
  euler.set_timesteps(50);
  ab2.set_timesteps(50);
  ab2.set_sampler(SamplerKind::kAb2);

  const std::vector<float> x = {1.0f, -2.0f, 0.5f};
  const std::vector<float> v = {0.25f, 1.0f, -0.5f};
  std::vector<float> oe(3), oa(3);
  euler.step(0, x.data(), v.data(), 3, oe.data());
  ab2.step(0, x.data(), v.data(), 3, oa.data());
  CHECK_CLOSE(oe, oa, 0.0, "ab2 step 0 is Euler");

  // And the second step is not, once there is a history to extrapolate from.
  const std::vector<float> v1 = {0.5f, 0.25f, -1.0f};
  euler.step(1, oe.data(), v1.data(), 3, oe.data());
  ab2.step(1, oa.data(), v1.data(), 3, oa.data());
  bool differs = false;
  for (int j = 0; j < 3; ++j) {
    if (oe[static_cast<size_t>(j)] != oa[static_cast<size_t>(j)]) differs = true;
  }
  CHECK_MSG(differs, "ab2 step 1 matched Euler on a changing velocity; it is not extrapolating");

  // Explicitly: step 1 is Euler applied to v1 + 0.5*(v1 - v0), which is the
  // 1.5/-0.5 extrapolation associated the way scheduler.cpp associates it.
  const std::vector<float> v_hat = {v1[0] + 0.5f * (v1[0] - v[0]), v1[1] + 0.5f * (v1[1] - v[1]),
                                    v1[2] + 0.5f * (v1[2] - v[2])};
  std::vector<float> want = {x[0], x[1], x[2]};
  {
    FlowScheduler plain(12.0f);
    plain.set_timesteps(50);
    want = reference_euler_step(plain, 0, want, v);
    want = reference_euler_step(plain, 1, want, v_hat);
  }
  CHECK_CLOSE(want, oa, 0.0, "ab2 is Euler with the extrapolated velocity substituted");

  // And that association is the literal 1.5/-0.5 form to fp32 rounding, so the
  // coefficients really are AB2's and not something adjacent.
  const std::vector<float> literal = {1.5f * v1[0] - 0.5f * v[0], 1.5f * v1[1] - 0.5f * v[1],
                                      1.5f * v1[2] - 0.5f * v[2]};
  CHECK_CLOSE(literal, v_hat, 1e-6, "v + 0.5*(v - v_prev) is 1.5*v - 0.5*v_prev");
}

// The accuracy claim. On dx/dsigma = k*x down the real shifted grid, AB2 must
// be markedly closer to exp(-k) than Euler, and must converge faster as the
// grid is refined. The grid is non-uniform — sigma is shifted by 12 or 3 — so
// the fixed 1.5/-0.5 coefficients are the constant-step formula on a variable
// step. That costs accuracy but not the order: refining the grid shrinks
// (h_n - h_{n-1}) as h^2, so the local error stays O(h^3).
VIDFAB_TEST(ab2_is_second_order_on_a_linear_ode) {
  const float k = 1.0f;
  const double exact = std::exp(-static_cast<double>(k));

  const int kGrids[] = {13, 25, 49, 97};
  for (const float shift : {12.0f, 3.0f}) {
    double e_err[4] = {0, 0, 0, 0};
    double a_err[4] = {0, 0, 0, 0};
    for (int g = 0; g < 4; ++g) {
      e_err[g] = std::abs(linear_ode(shift, kGrids[g], SamplerKind::kEuler, k) - exact);
      a_err[g] = std::abs(linear_ode(shift, kGrids[g], SamplerKind::kAb2, k) - exact);
      std::printf("  shift %4.0f  %3d points  euler %.4e  ab2 %.4e  ratio %5.2f\n",
                  static_cast<double>(shift), kGrids[g], e_err[g], a_err[g],
                  e_err[g] / a_err[g]);
    }

    for (int g = 0; g < 4; ++g) {
      CHECK_MSG(a_err[g] < e_err[g], "shift %.0f, %d points: ab2 error %.3e is not below euler's %.3e",
                static_cast<double>(shift), kGrids[g], a_err[g], e_err[g]);
      // Past the coarsest grid the margin is a factor, not a nudge. The
      // coarsest is left out on purpose: at 12 steps the shift-12 grid's step
      // ratio is far enough from 1 that the fixed coefficients give up most of
      // their advantage, which is itself worth knowing.
      if (g > 0) {
        CHECK_MSG(a_err[g] < e_err[g] * 0.4,
                  "shift %.0f, %d points: ab2 error %.3e is not 2.5x better than euler's %.3e",
                  static_cast<double>(shift), kGrids[g], a_err[g], e_err[g]);
      }
    }

    // The order. Halving the step halves a first-order method's error and
    // quarters a second-order one's. The grids double, so consecutive entries
    // are a halving; the bounds are loose enough for the pre-asymptotic end
    // and tight enough that a first-order method cannot pass the ab2 one.
    for (int g = 1; g < 4; ++g) {
      const double euler_rate = e_err[g - 1] / e_err[g];
      const double ab2_rate = a_err[g - 1] / a_err[g];
      CHECK_MSG(euler_rate > 1.7 && euler_rate < 2.4,
                "shift %.0f, %d points: euler converged at %.2fx per halving, expected ~2",
                static_cast<double>(shift), kGrids[g], euler_rate);
      CHECK_MSG(ab2_rate > 3.0,
                "shift %.0f, %d points: ab2 converged at %.2fx per halving, expected ~4; it is "
                "not second order",
                static_cast<double>(shift), kGrids[g], ab2_rate);
    }

    // The claim the whole track rests on, stated as an assertion: ab2 on half
    // the evaluations is *more* accurate than euler on all of them. True here
    // for a linear ODE, which is the easiest possible case and says nothing
    // about a distilled checkpoint.
    //
    // It starts at 25 points rather than 13, and the exception is the useful
    // part: at shift 12 with 12 steps, ab2's 4.07e-2 does *not* beat euler's
    // 3.79e-2 on 24. The shift-12 grid's step ratio is far enough from 1 there
    // that the fixed 1.5/-0.5 coefficients lose most of their order. Halving
    // an already short schedule is exactly where this sampler has least to
    // offer, which is worth carrying into any --steps recommendation.
    for (int g = 2; g < 4; ++g) {
      CHECK_MSG(a_err[g - 1] < e_err[g],
                "shift %.0f: ab2 on %d points (%.3e) did not beat euler on %d (%.3e)",
                static_cast<double>(shift), kGrids[g - 1], a_err[g - 1], kGrids[g], e_err[g]);
    }
  }
}

// The house habit: compute the plausible wrong forms too and require that the
// implementation does not match them. Every one of these produces a finite,
// correctly shaped, roughly right trajectory.
VIDFAB_TEST(ab2_rejects_plausible_wrong_extrapolations) {
  const float k = 1.0f;
  const int grid = 25;
  const double exact = std::exp(-static_cast<double>(k));
  const double ours = linear_ode(12.0f, grid, SamplerKind::kAb2, k);
  const double ours_err = std::abs(ours - exact);

  struct Wrong {
    const char* name;
    float c_now;
    float c_prev;
  };
  const Wrong wrong[] = {
      {"coefficients swapped (-0.5*v_n + 1.5*v_prev)", -0.5f, 1.5f},
      {"history term added rather than subtracted", 1.5f, 0.5f},
      {"AB2 for the AM2 corrector (0.5/0.5)", 0.5f, 0.5f},
      {"a whole-step sign flip", -1.5f, 0.5f},
  };
  for (const Wrong& w : wrong) {
    const double bad = linear_ode_wrong(12.0f, grid, k, w.c_now, w.c_prev);
    CHECK_MSG(std::abs(bad - ours) > 1e-6,
              "ab2 matched the wrong form '%s' (%.9f vs %.9f)", w.name, bad, ours);
    CHECK_MSG(std::abs(bad - exact) > ours_err,
              "the wrong form '%s' integrated the ODE better than ab2 did (%.3e vs %.3e)", w.name,
              std::abs(bad - exact), ours_err);
  }
}

// Euler must be exactly what it was. This is the gate on the whole track: a
// mode was added, the default was not perturbed. Full trajectories at both
// live shifts, compared at zero tolerance against a transcription of the
// pre-change update.
VIDFAB_TEST(euler_mode_is_bit_identical_to_the_reference_update) {
  for (const float shift : {12.0f, 3.0f}) {
    for (const int grid : {6, 30, 50}) {
      FlowScheduler s(shift);
      s.set_timesteps(grid);
      CHECK(s.sampler() == SamplerKind::kEuler);

      std::vector<float> x = ::vidfab::test::make_data(64, 0x51ED + static_cast<uint32_t>(grid));
      std::vector<float> want = x;
      const int steps = static_cast<int>(s.num_steps());
      for (int i = 0; i < steps; ++i) {
        // A state-dependent velocity, so any divergence compounds instead of
        // cancelling.
        std::vector<float> v(want.size());
        for (size_t j = 0; j < v.size(); ++j) v[j] = 0.3f * want[j] - 0.05f;
        want = reference_euler_step(s, i, want, v);
        s.step(i, x.data(), v.data(), x.size(), x.data());
      }
      CHECK_CLOSE(want, x, 0.0, "euler trajectory is bitwise the pre-change update");
    }
  }
}

// The two schedulers a request runs — shift 12 for video, shift 3 for audio —
// are stepped independently inside one loop iteration, at different buffer
// lengths. Each must carry its own history.
VIDFAB_TEST(ab2_histories_do_not_cross_between_schedulers) {
  FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(20);
  audio.set_timesteps(20);
  video.set_sampler(SamplerKind::kAb2);
  audio.set_sampler(SamplerKind::kAb2);
  CHECK(video.num_steps() == audio.num_steps());

  std::vector<float> xv = ::vidfab::test::make_data(37, 0x1234);
  std::vector<float> xa = ::vidfab::test::make_data(11, 0x5678);
  const std::vector<float> xv0 = xv, xa0 = xa;

  const int steps = static_cast<int>(video.num_steps());
  auto vel = [](const std::vector<float>& x, int i, float g) {
    std::vector<float> v(x.size());
    for (size_t j = 0; j < v.size(); ++j) v[j] = g * x[j] + 0.01f * static_cast<float>(i);
    return v;
  };
  for (int i = 0; i < steps; ++i) {
    const std::vector<float> vv = vel(xv, i, 0.4f);
    const std::vector<float> va = vel(xa, i, -0.7f);
    video.step(i, xv.data(), vv.data(), xv.size(), xv.data());
    audio.step(i, xa.data(), va.data(), xa.size(), xa.data());
  }

  // The same two trajectories run one after the other rather than interleaved.
  FlowScheduler lone_video(12.0f), lone_audio(3.0f);
  lone_video.set_timesteps(20);
  lone_audio.set_timesteps(20);
  lone_video.set_sampler(SamplerKind::kAb2);
  lone_audio.set_sampler(SamplerKind::kAb2);
  std::vector<float> yv = xv0, ya = xa0;
  for (int i = 0; i < steps; ++i) {
    const std::vector<float> vv = vel(yv, i, 0.4f);
    lone_video.step(i, yv.data(), vv.data(), yv.size(), yv.data());
  }
  for (int i = 0; i < steps; ++i) {
    const std::vector<float> va = vel(ya, i, -0.7f);
    lone_audio.step(i, ya.data(), va.data(), ya.size(), ya.data());
  }
  CHECK_CLOSE(yv, xv, 0.0, "interleaved video history");
  CHECK_CLOSE(ya, xa, 0.0, "interleaved audio history");
}

// The wiring, which the cross-build bit-identity proof cannot see.
//
// That proof compiles two versions of scheduler.cpp against a driver that
// never mentions SamplerKind, so it pins the *scheduler's* default path
// exactly. What it cannot reach is the path from the CLI to the scheduler:
// `RunOptions::sampler` and the two `set_sampler` calls `run_generate` makes.
// If that field defaulted to kAb2, or the call were dropped, the md5 proof
// would still pass and every euler run would silently be something else.
//
// It needs no GPU: RunOptions is a plain struct in a host-safe header, and
// what a scheduler does with the value it is handed is host arithmetic. The
// CLI's own string parsing lives inside main.cpp and is not linkable, so it
// was verified by execution instead: `--sampler ab2 --dry-run` resolves, and
// `--sampler nonsense` exits 2 naming the valid values.
VIDFAB_TEST(run_options_default_selects_the_euler_path) {
  // The default a run gets when no --sampler flag is passed at all.
  const vidfab::RunOptions defaults;
  CHECK(defaults.sampler == SamplerKind::kEuler);

  // And handing that default to a scheduler must be indistinguishable from
  // never touching the sampler — bitwise, over a whole trajectory. This is the
  // assertion that fails if the default ever moves.
  const std::vector<float> x0 = ::vidfab::test::make_data(48, 0xE01E);
  auto trajectory = [&](bool configure, SamplerKind kind) {
    FlowScheduler s(12.0f);
    s.set_timesteps(30);
    if (configure) s.set_sampler(kind);
    std::vector<float> x = x0;
    const int steps = static_cast<int>(s.num_steps());
    for (int i = 0; i < steps; ++i) {
      std::vector<float> v(x.size());
      for (size_t j = 0; j < v.size(); ++j) v[j] = 0.3f * x[j] - 0.05f;
      s.step(i, x.data(), v.data(), x.size(), x.data());
    }
    return x;
  };

  const std::vector<float> untouched = trajectory(false, SamplerKind::kEuler);
  const std::vector<float> via_defaults = trajectory(true, defaults.sampler);
  CHECK_CLOSE(untouched, via_defaults, 0.0,
              "RunOptions' default drives the same path as never calling set_sampler");

  // Explicit euler is the same path too, so `--sampler euler` is not a
  // different mode that merely looks like the default.
  const std::vector<float> explicit_euler = trajectory(true, SamplerKind::kEuler);
  CHECK_CLOSE(untouched, explicit_euler, 0.0, "--sampler euler is the default path");

  // And the plumbing is not a no-op: asking for ab2 must actually change the
  // trajectory, or a dropped set_sampler call would pass everything above.
  const std::vector<float> as_ab2 = trajectory(true, SamplerKind::kAb2);
  bool differs = false;
  for (size_t j = 0; j < as_ab2.size(); ++j) {
    if (as_ab2[j] != untouched[j]) differs = true;
  }
  CHECK_MSG(differs, "set_sampler(kAb2) left the trajectory unchanged; the plumbing is dead");
}

// State makes ordering load-bearing, so the errors have to be exceptions and
// not a silently wrong extrapolation from some other point of the trajectory.
VIDFAB_TEST(ab2_step_rejects_disordered_and_mismatched_calls) {
  // Skipping an index.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> x = {1.0f}, v = {0.5f};
    s.step(0, x.data(), v.data(), 1, x.data());
    s.step(2, x.data(), v.data(), 1, x.data());
  }));

  // Repeating one.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> x = {1.0f}, v = {0.5f};
    s.step(0, x.data(), v.data(), 1, x.data());
    s.step(1, x.data(), v.data(), 1, x.data());
    s.step(1, x.data(), v.data(), 1, x.data());
  }));

  // Going backwards.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> x = {1.0f}, v = {0.5f};
    s.step(0, x.data(), v.data(), 1, x.data());
    s.step(1, x.data(), v.data(), 1, x.data());
    s.step(2, x.data(), v.data(), 1, x.data());
    s.step(1, x.data(), v.data(), 1, x.data());
  }));

  // A restart at 0 does not license the next index being wrong either.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> x = {1.0f}, v = {0.5f};
    s.step(0, x.data(), v.data(), 1, x.data());
    s.step(1, x.data(), v.data(), 1, x.data());
    s.step(0, x.data(), v.data(), 1, x.data());  // legal: index 0 restarts
    s.step(2, x.data(), v.data(), 1, x.data());  // not legal: 1 was expected
  }));

  // Changing the buffer length mid-trajectory. Legal between trajectories,
  // meaningless within one.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> v = {0.1f, 0.2f, 0.3f, 0.4f};
    s.step(0, x.data(), v.data(), 4, x.data());
    s.step(1, x.data(), v.data(), 2, x.data());
  }));

  // The range check that was already there still fires.
  CHECK(::vidfab::test::throws([] {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    std::vector<float> x = {1.0f}, v = {0.5f};
    s.step(9999, x.data(), v.data(), 1, x.data());
  }));

  // Index 0 restarts, which is what lets one scheduler serve several runs —
  // the denoising loop reuses its two across calls. A restart must be a true
  // restart: first-order again, no history from the previous trajectory.
  {
    FlowScheduler s(12.0f);
    s.set_timesteps(10);
    s.set_sampler(SamplerKind::kAb2);
    std::vector<float> a = {1.0f}, b = {1.0f};
    const std::vector<float> v0 = {0.5f}, v1 = {-0.25f};
    s.step(0, a.data(), v0.data(), 1, a.data());
    s.step(1, a.data(), v1.data(), 1, a.data());
    // Second trajectory, same inputs, on the same object.
    s.step(0, b.data(), v0.data(), 1, b.data());
    s.step(1, b.data(), v1.data(), 1, b.data());
    CHECK_CLOSE(a, b, 0.0, "a restart at index 0 forgets the previous trajectory");
  }

  // set_timesteps, set_sampler and reset all clear the history. Probed at a
  // *non-zero* index, because index 0 restarts on its own and would pass this
  // even if none of them cleared anything: after a clear the cursor accepts
  // any index as the first of a fresh trajectory, so step(1) here has to be
  // first-order. If v_{n-1} survived, it would extrapolate and diverge.
  {
    const std::vector<float> v0 = {0.5f}, v1 = {-0.25f}, v2 = {0.125f};

    // What a first-order step at index 1 looks like, from a scheduler that has
    // never stepped at all.
    FlowScheduler fresh(12.0f);
    fresh.set_timesteps(10);
    fresh.set_sampler(SamplerKind::kAb2);
    std::vector<float> want = {1.0f};
    fresh.step(1, want.data(), v2.data(), 1, want.data());

    auto probe = [&](void (*clear)(FlowScheduler&), const char* what) {
      FlowScheduler s(12.0f);
      s.set_timesteps(10);
      s.set_sampler(SamplerKind::kAb2);
      std::vector<float> x = {1.0f};
      s.step(0, x.data(), v0.data(), 1, x.data());
      s.step(1, x.data(), v1.data(), 1, x.data());  // history now holds v1
      clear(s);
      x[0] = 1.0f;
      s.step(1, x.data(), v2.data(), 1, x.data());
      CHECK_CLOSE(want, x, 0.0, what);
    };
    probe([](FlowScheduler& s) { s.set_timesteps(10); }, "set_timesteps clears the history");
    probe([](FlowScheduler& s) { s.set_sampler(SamplerKind::kAb2); },
          "set_sampler clears the history");
    probe([](FlowScheduler& s) { s.reset(); }, "reset clears the history");
  }
}
