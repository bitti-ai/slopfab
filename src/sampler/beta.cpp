#include "slopfab/sampler/scheduler.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::sampler {
namespace {
// Continued fraction for the regularized incomplete beta function (Lentz).
double fraction(double a, double b, double x) {
  constexpr double tiny = 1e-300;
  double c = 1, d = 1 - (a + b) * x / (a + 1);
  if (std::abs(d) < tiny) d = tiny;
  d = 1 / d;
  double h = d;
  for (int m = 1; m <= 500; ++m) {
    const double mm = m;
    double aa = mm * (b - mm) * x / ((a + 2 * mm - 1) * (a + 2 * mm));
    for (int half = 0; half < 2; ++half) {
      if (half) aa = -(a + mm) * (a + b + mm) * x / ((a + 2 * mm) * (a + 2 * mm + 1));
      d = 1 + aa * d; if (std::abs(d) < tiny) d = tiny;
      c = 1 + aa / c; if (std::abs(c) < tiny) c = tiny;
      d = 1 / d;
      const double delta = d * c;
      h *= delta;
      if (half && std::abs(delta - 1) < 3e-14) return h;
    }
  }
  throw std::runtime_error("beta CDF failed to converge");
}

double cdf(double x, double a, double b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const double factor = std::exp(std::lgamma(a+b)-std::lgamma(a)-std::lgamma(b)
                                 + a*std::log(x)+b*std::log1p(-x));
  if (x < (a+1)/(a+b+2)) return factor*fraction(a,b,x)/a;
  return 1-factor*fraction(b,a,1-x)/b;
}

double quantile(double p, double a, double b) {
  if (p >= 1) return 1;
  if (p == .5 && a == b) return .5;
  double lo=0, hi=1;
  for (int i=0;i<64;++i) {
    double mid=(lo+hi)*.5;
    if(cdf(mid,a,b)<p) lo=mid; else hi=mid;
  }
  return (lo+hi)*.5;
}
}

std::vector<float> beta_sigma_schedule(int steps, float shift, float alpha, float beta,
                                       int training_steps) {
  if (steps < 1 || steps > 1000000 || training_steps < 2 || training_steps > 1000000 ||
      !std::isfinite(shift) || shift <= 0 || !std::isfinite(alpha) || !std::isfinite(beta) ||
      alpha < .01f || beta < .01f || alpha > 100 || beta > 100)
    throw std::invalid_argument("beta schedule requires positive shift, 1..1000000 steps and alpha/beta in [0.01,100]");
  std::vector<float> sigmas;
  for (int i=0;i<steps;++i) {
    const double position=quantile(1-double(i)/steps,alpha,beta)*(training_steps-1);
    const double floor=std::floor(position), part=position-floor;
    // NumPy rint uses round-to-nearest, ties-to-even, independently of fenv.
    const int index=int(floor)+(part>.5 || (part==.5 && int(floor)%2));
    const double t=double(index+1)/training_steps;
    const float sigma=float(shift*t/(1+(shift-1.)*t));
    if(sigmas.empty() || sigma < sigmas.back()) sigmas.push_back(sigma);
  }
  sigmas.push_back(0);
  return sigmas;
}
}
