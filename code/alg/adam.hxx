#ifndef _glia_alg_adam_hxx_
#define _glia_alg_adam_hxx_

#include "type/optimizer.hxx"
#include "util/container.hxx"

namespace glia {
namespace opt {

template <typename TFunc>
class TAdamOptimizer : public TOptimizer<TFunc> {
 public:
  typedef TOptimizer<TFunc> Super;
  typedef TAdamOptimizer<TFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TFunc Function;

  struct AdamParameter {
    double step = 0.01;
    double stepDecay = 0.8;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double gradNormTolerance = 1e-16;
    int maxIterations = 20;
  };

  AdamParameter param;

  TAdamOptimizer (
      std::shared_ptr<Function> pFunc,
      std::shared_ptr<typename Function::Input> pInput,
      bool displayProgress, double step, double stepDecay,
      int maxIterations) : Super(pFunc, pInput, displayProgress) {
    param.step = step;
    param.stepDecay = stepDecay;
    param.maxIterations = maxIterations;
  }

  ~TAdamOptimizer () override {}

  void run (bool reset) override { fixedStepAdam(); }

 protected:
  double helperEvaluateAll () {
    Super::pInput->turnOnUseAll();
    double fx = Super::pFunc->operator()(*Super::pInput);
    Super::pInput->turnOffUseAll();
    return fx;
  }

  double helperEvaluateAll (double* g) {
    Super::pInput->turnOnUseAll();
    double fx = Super::pFunc->operator()(g, *Super::pInput);
    Super::pInput->turnOffUseAll();
    return fx;
  }

  void helperUpdate (double* px, double const* pg, double step) {
    int d = Super::pFunc->dim();
    double beta1C = 1.0 - beta1;
    double beta2C = 1.0 - beta2;
    // m = beta1 * m + (1 - beta1) * g
    double beta1C = 1.0 - beta1;
    unary_op(_m.get(), pg, d, [this, beta1C](
        double& m, double const& g) {
               m = this->beta1 * m + beta1C * g});
    // v = beta2 * v + (1 - beta2) * g .^ 2
    double beta2C = 1.0 - beta2;
    unary_op(_v.get(), pg, d, [this, beta2C](
        double& v, double const& g) {
               v = this->beta2 * v + beta2C * g * g; });
    // x = x - step * m / sqrt(v + eps)
    binary_op(px, _m.get(), _v.get(), d, [this](
        double& x, double const& m, double const& v) {
                x -= step * m / std::sqrt(v + this->eps); });
    Super::pFunc->update(px, d);
  }

  // Assume actual x is identical to _x
  void helperFullBatchGD (double step)
  { helperUpdate(_x.get(), _g.get(), step); }

  // Assume actual x is identical to _x
  void helperMiniBatchGD (double step) {  // Only run for one batch
    int d = Super::pFunc->dim();
    if (param.nEpochPerIteration <= 0) {
      Super::pInput->prepareBatch();
      Super::pFunc->operator()(_g_bk, get(), *Super::pInput);
      helperUpdate(_x.get(), _g_bk.get(), step);
    } else {  // Run for epoch(s)
      BatchSampler::BatchType batchType;
      for (int ep = 0; ep < param.nEpochPerIteration; ++ep) {
        do {
          batchType = Super::pInput->prepareBatch();
          Super::pFunc->operator()(_g_bk.get(), *Super::pInput);
          helperUpdate(_x.get(), _g_bk.get(), step);
        } while (batchType != BatchSampler::BatchType::EndOfEpoch);
      }
    }
  }

  void fixedStepGD () {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    bool isUsingBatchSampler = Super::pInput->isUsingBatchSampler();
    if (!_x) { _x.reset(new double[d]); }
    if (!_g) { _g.reset(new double[d]); }
    if (!_m) { _m.reset(new double[d]); }
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_m.get(), d, 0.0);
    std::fill_n(_v.get(), d, 0.0);
    if (isUsingBatchSampler && !_g_bk) { _g_bk.reset(new double[d]); }
    Super::pFunc->copy(_x.get());
    for (int t = 0; param.maxIterations < 0 || t < param.maxIterations;
         ++t) {
      // Stats before taking a step
      double fx = helperEvaluateAll(_g.get());
      double gnorm = std::sqrt(opSquaredNorm(_g.get(), d));
      if (Super::displayProgress) {
        double xnorm = std::sqrt(opSquaredNorm(_x.get(), d));
        printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g "
               "step = %g\n", t, fx, xnorm, gnorm, param.step);
      }
      // Stop if gradient is small
      if (gnorm < param.gradNormTolerance) { break; }
      // Take a step/epoch
      if (isUsingBatchSampler) { helperMiniBatchGD(param.step); }
      else { helperFullBatchGD(param.step); }
    }
  }

  void adaptiveStepGD (bool reset) {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    bool isUsingBatchSampler = Super::pInput->isUsingBatchSampler();
    if (!_x_bk) { _x_bk.reset(new double[d]); }
    if (!_x) { _x.reset(new double[d]); }
    if (!_g) { _g.reset(new double[d]); }
    if (!_m) { _m.reset(new double[d]); }
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_m.get(), d, 0.0);
    std::fill_n(_v.get(), d, 0.0);
    if (isUsingBatchSampler && !_g_bk) { _g_bk.reset(new double[d]); }
    double step = reset ? param.step : prevStep;
    int t = 0;
    double fx = FMAX;
    Super::pFunc->copy(_x.get());
    while (param.maxIterations < 0 || t < param.maxIterations) {
      // Stop if gradient for all samples is small
      fx = helperEvaluateAll(_g.get());
      double gnorm = std::sqrt(opSquaredNorm(_g.get(), d));
      if (gnorm < param.gradNormTolerance) { break; }
      // Back up
      Super::pFunc->copy(_x_bk.get());
      // Stats before taking a step
      if (Super::displayProgress) {
        double xnorm = std::sqrt(opSquaredNorm(_x_bk.get(), d));
        printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g ",
               t, fx, xnorm, gnorm);
      }
      // Take a step and evaluate
      if (isUsingBatchSampler) { helperMiniBatchGD(step); }
      else { helperFullBatchGD(step); }
      double fx_prev = fx;
      fx = helperEvaluateAll();
      while (fx >= 0.95 * fx_prev) {
        // Shrink step
        step *= param.stepDecay;
        if (step < param.minStep) { break; }
        // Roll back, take a step, and re-evaluate
        Super::pFunc->update(_x_bk.get(), d);
        Super::pFunc->copy(_x.get());
        if (isUsingBatchSampler) { helperMiniBatchGD(step); }
        else { helperFullBatchGD(step); }
        fx = helperEvaluateAll();
      }
      // If fail to find valid step, roll back and break
      if (step < param.minStep) {
        fx = fx_prev;
        Super::pFunc->update(_x_bk.get(), d);
        if (Super::displayProgress) { printf("step = n/a\n"); }
        ++t; break;
      } else if (Super::displayProgress) { printf("step = %g\n", step); }
      // Increase iteration
      ++t;
    }
    // Final stats
    if (Super::displayProgress) {
      double gnorm = std::sqrt(opSquaredNorm(_g.get(), d));
      Super::pFunc->copy(_x.get());
      double xnorm = std::sqrt(opSquaredNorm(_x.get(), d));
      printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g step = %g\n",
             t, fx, xnorm, gnorm, step);
    }
    // Keep step as previous step
    prevStep = step;
  }

  std::unique_ptr<double> _x;
  std::unique_ptr<double> _x_bk;
  std::unique_ptr<double> _g;
  std::unique_ptr<double> _g_bk;
  std::unique_ptr<double> _m;
  std::unique_ptr<double> _v;
};

};
};

#endif
