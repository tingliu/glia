#ifndef _glia_alg_gd_hxx_
#define _glia_alg_gd_hxx_

#include "type/optimizer.hxx"
#include "util/container.hxx"

namespace glia {
namespace opt {

template <typename TFunc>
class TGdOptimizer : public TOptimizer<TFunc> {
 public:
  typedef TOptimizer<TFunc> Super;
  typedef TGdOptimizer<TFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TFunc Function;

  struct GdParameter {
    double step = 0.01;
    double stepDecay = 0.8;  // Use 1.0 to skip adaptive step size
    int fixedStepDecayIterations = -1;
    double fixedStepDecayRate = 0.5;
    double minStep = 1e-10;
    double gradNormTolerance = 1e-16;
    int maxIterations = 20;
    int nEpochPerIteration = 1;
    int nBatchPerIteration = 1;
    // Whether to alternately take steps
    // Assume both terms are initially turned on
    bool altSU = false;
  };

  GdParameter param;

  TGdOptimizer (
      std::shared_ptr<Function> pFunc,
      std::shared_ptr<typename Function::Input> pInput,
      bool displayProgress, double step, double stepDecay,
      int fixedStepDecayIterations, int maxIterations,
      int nEpochPerIteration, int nBatchPerIteration)
      : Super(pFunc, pInput, displayProgress) {
    param.step = step;
    param.stepDecay = stepDecay;
    param.fixedStepDecayIterations = fixedStepDecayIterations;
    param.maxIterations = maxIterations;
    param.nEpochPerIteration = nEpochPerIteration;
    param.nBatchPerIteration = nBatchPerIteration;
  }

  ~TGdOptimizer () override {}

  void run () override {
    if (isfeq(param.stepDecay, 1.0)) { fixedStepGD(); }
    else { adaptiveStepGD(); }
  }

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

  virtual void helperUpdate (
      double* px, double const* pg, double step) {
    int d = Super::pFunc->dim();
    unary_op(px, pg, d, [step](double& x, double const& g) {
        x -= g * step; });
    Super::pFunc->update(px, d);
  }

  // Assume actual x is identical to _x[0]
  virtual void helperFullBatchGD (double step)
  { helperUpdate(_x[0].get(), _g[0].get(), step); }

  // Assume actual x is identical to _x[0]
  virtual void helperMiniBatchGD (double step) {
    int d = Super::pFunc->dim();
    if (param.nEpochPerIteration > 0) {  // Run for epoch(s)
      BatchSampler::BatchType batchType;
      for (int ep = 0; ep < param.nEpochPerIteration; ++ep) {
        do {
          batchType = Super::pInput->prepareBatch();
          Super::pFunc->operator()(_g[1].get(), *Super::pInput);
          // // Debug
          // double fx = Super::pFunc->operator()(
          //     _g[1].get(), *Super::pInput);
          // double gnorm = std::sqrt(opSquaredNorm(_g[1].get(), d));
          // double xnorm = std::sqrt(opSquaredNorm(_x[0].get(), d));
          // printf("\tep = %-6d batch: %-6d fx = %-12g |x| = %-12g "
          //        "|g| = %-12g step = %g\n", ep,
          //        Super::pInput->unsupervised->size(), fx, xnorm, gnorm,
          //        step);
          // // ~ Debug
          helperUpdate(_x[0].get(), _g[1].get(), step);
        } while (batchType != BatchSampler::BatchType::EndOfEpoch);
      }
    } else {  // Run for certain number of batches
      bool origUseSupervised = Super::pFunc->useSupervised;
      bool origUseUnsupervised = Super::pFunc->useUnsupervised;
      for (int i = 0; i < param.nBatchPerIteration; ++i) {
        if (origUseSupervised && origUseUnsupervised && param.altSU) {
          if (i == 0) {
            Super::pFunc->useSupervised = true;
            Super::pFunc->useUnsupervised = false;
            Super::pInput->useSupervised = true;
            Super::pInput->useUnsupervised = false;
          } else {
            Super::pFunc->useSupervised =
                !Super::pFunc->useSupervised;
            Super::pFunc->useUnsupervised =
                !Super::pFunc->useUnsupervised;
            Super::pInput->useSupervised =
                !Super::pInput->useSupervised;
            Super::pInput->useUnsupervised =
                !Super::pInput->useUnsupervised;
          }
        }
        Super::pInput->prepareBatch();
        double fx = Super::pFunc->operator()(_g[1].get(), *Super::pInput);
        // // Debug
        // double gnorm = std::sqrt(opSquaredNorm(_g[1].get(), d));
        // double xnorm = std::sqrt(opSquaredNorm(_x[0].get(), d));
        // if (Super::pFunc->useSupervised) {
        //   disp(
        //       "\t\tbi = %-6d [sup] batch: %-6d fx = %-12g "
        //       "|x| = %-12g |g| = %-12g step = %g", i,
        //       Super::pInput->supervised->size(), fx, xnorm, gnorm,
        //       step);
        // }
        // if (Super::pFunc->useUnsupervised) {
        //   disp(
        //       "\t\tbi = %-6d [uns] batch: %-6d fx = %-12g |x| = %-12g "
        //       "|g| = %-12g step = %g", i,
        //       Super::pInput->unsupervised->size(), fx, xnorm, gnorm,
        //       step);
        // }
        // // ~ Debug
        helperUpdate(_x[0].get(), _g[1].get(), step);
      }
      if (origUseSupervised && origUseUnsupervised && param.altSU) {
        Super::pFunc->useSupervised = true;
        Super::pFunc->useUnsupervised = true;
        Super::pInput->useSupervised = true;
        Super::pInput->useUnsupervised = true;
      }
    }
  }

  virtual void fixedStepGD () {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    bool isUsingBatchSampler = Super::pInput->isUsingBatchSampler();
    _x.resize(2);
    _g.resize(2);
    if (!_x[0]) { _x[0].reset(new double[d]); }
    if (!_g[0]) { _g[0].reset(new double[d]); }
    if (isUsingBatchSampler && !_g[1]) { _g[1].reset(new double[d]); }
    Super::pFunc->copy(_x[0].get());
    double step = param.step;
    for (int t = 0; param.maxIterations < 0 || t < param.maxIterations;
         ++t) {
      if (param.fixedStepDecayIterations > 0 && t > 0 &&
          t % param.fixedStepDecayIterations == 0)
      { step = std::max(param.minStep, step * param.fixedStepDecayRate); }
      // Stats before taking a step
      double fx = helperEvaluateAll(_g[0].get());
      double gnorm = std::sqrt(opSquaredNorm(_g[0].get(), d));
      if (Super::displayProgress) {
        double xnorm = std::sqrt(opSquaredNorm(_x[0].get(), d));
        // // Trial
        // printf(
        //     "nNonZeroLoss = %d ",
        //     Super::pFunc->unsupervised->nNonZeroLoss);
        // // ~ Trial
        printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g "
               "step = %g\n", t, fx, xnorm, gnorm, step);
      }
      // Stop if gradient is small
      if (gnorm < param.gradNormTolerance) { break; }
      // Take a step/epoch
      if (isUsingBatchSampler) { helperMiniBatchGD(step); }
      else { helperFullBatchGD(step); }
    }
    // Final stats
    if (Super::displayProgress) {
      double fx = helperEvaluateAll(_g[0].get());
      double gnorm = std::sqrt(opSquaredNorm(_g[0].get(), d));
      Super::pFunc->copy(_x[0].get());
      double xnorm = std::sqrt(opSquaredNorm(_x[0].get(), d));
      // // Trial
      // printf(
      //     "nNonZeroLoss = %d ",
      //     Super::pFunc->unsupervised->nNonZeroLoss);
      // // ~ Trial
      printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g step = %g\n",
            param.maxIterations, fx, xnorm, gnorm, step);
    }
  }

  virtual void adaptiveStepGD () {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    bool isUsingBatchSampler = Super::pInput->isUsingBatchSampler();
    _x.resize(2);
    _g.resize(2);
    if (!_x[1]) { _x[1].reset(new double[d]); }
    if (!_x[0]) { _x[0].reset(new double[d]); }
    if (!_g[0]) { _g[0].reset(new double[d]); }
    if (isUsingBatchSampler && !_g[1]) { _g[1].reset(new double[d]); }
    double step = param.step;
    int t = 0;
    double fx = FMAX;
    Super::pFunc->copy(_x[0].get());
    while (param.maxIterations < 0 || t < param.maxIterations) {
      // Stop if gradient for all samples is small
      fx = helperEvaluateAll(_g[0].get());
      double gnorm = std::sqrt(opSquaredNorm(_g[0].get(), d));
      if (gnorm < param.gradNormTolerance) { break; }
      // Back up
      Super::pFunc->copy(_x[1].get());
      // Stats before taking a step
      if (Super::displayProgress) {
        double xnorm = std::sqrt(opSquaredNorm(_x[1].get(), d));
        printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g ",
               t, fx, xnorm, gnorm);
      }
      // Take a step and evaluate
      if (isUsingBatchSampler) { helperMiniBatchGD(step); }
      else { helperFullBatchGD(step); }
      double fx_prev = fx;
      fx = helperEvaluateAll();
      while (step > param.minStep && fx >= fx_prev) {
        // Shrink step
        step = std::max(param.minStep, step * param.stepDecay);
        if (step == param.minStep) { break; }
        // Roll back, take a step, and re-evaluate
        Super::pFunc->update(_x[1].get(), d);
        Super::pFunc->copy(_x[0].get());
        if (isUsingBatchSampler) { helperMiniBatchGD(step); }
        else { helperFullBatchGD(step); }
        fx = helperEvaluateAll();
      }
      // // If fail to find valid step, roll back and break
      // if (step < param.minStep) {
      //   fx = fx_prev;
      //   Super::pFunc->update(_x[1].get(), d);
      //   if (Super::displayProgress) { printf("step = n/a\n"); }
      //   ++t; break;
      // } else
      if (Super::displayProgress) { printf("step = %g\n", step); }
      // Increase iteration
      ++t;
    }
    // Final stats
    if (Super::displayProgress) {
      double gnorm = std::sqrt(opSquaredNorm(_g[0].get(), d));
      Super::pFunc->copy(_x[0].get());
      double xnorm = std::sqrt(opSquaredNorm(_x[0].get(), d));
      printf("\tgd: %-6d fx = %-12g |x| = %-12g |g| = %-12g step = %g\n",
             t, fx, xnorm, gnorm, step);
    }
  }

  std::vector<std::unique_ptr<double>> _x;
  std::vector<std::unique_ptr<double>> _g;
};


template <typename TFunc>
class TMomentumOptimizer : public TGdOptimizer<TFunc> {
 public:
  typedef TGdOptimizer<TFunc> Super;
  typedef TMomentumOptimizer<TFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TFunc Function;

  struct MomentumParameter : Super::GdParameter {
    double mu = 0.5;
  };

  MomentumParameter param;

  TMomentumOptimizer (
      std::shared_ptr<Function> pFunc,
      std::shared_ptr<typename Function::Input> pInput,
      bool displayProgress, double step, double stepDecay,
      int fixedStepDecayIterations, int maxIterations,
      int nEpochPerIteration, int nBatchPerIteration) : Super(
          pFunc, pInput, displayProgress, step, stepDecay,
          fixedStepDecayIterations, maxIterations, nEpochPerIteration,
          nBatchPerIteration) {}

  ~TMomentumOptimizer () override {}

 protected:
  void helperUpdate (double* px, double const* pg, double step) override {
    int d = Super::pFunc->dim();
    binary_op(px, pg, _v.get(), d, [this, step](
        double& x, double const& g, double const& v) {
                x -= step * (g + this->param.mu * v); });
    unary_op(_v.get(), pg, d, [this](double& v, double const& g) {
        v = g + this->param.mu * v; });
    Super::pFunc->update(px, d);
  }

  void fixedStepGD () override {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_v.get(), d, 0.0);
    Super::fixedStepGD();
  }

  void adaptiveStepGD () override {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_v.get(), d, 0.0);
    Super::adaptiveStepGD();
  }

  std::unique_ptr<double> _v;
};


template <typename TFunc>
class TAdamOptimizer : public TGdOptimizer<TFunc> {
 public:
  typedef TGdOptimizer<TFunc> Super;
  typedef TAdamOptimizer<TFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TFunc Function;

  struct AdamParameter : Super::GdParameter {
    double beta1 = 0.5;  // 0.99
    double beta2 = 0.9;  // 0.999
    double eps = 1e-8;
  };

  AdamParameter param;

  TAdamOptimizer (
      std::shared_ptr<Function> pFunc,
      std::shared_ptr<typename Function::Input> pInput,
      bool displayProgress, double step, double stepDecay,
      int fixedStepDecayIterations, int maxIterations,
      int nEpochPerIteration, int nBatchPerIteration) : Super(
          pFunc, pInput, displayProgress, step, stepDecay,
          fixedStepDecayIterations, maxIterations, nEpochPerIteration,
          nBatchPerIteration) {}

  ~TAdamOptimizer () override {}

 protected:
  void helperUpdate (double* px, double const* pg, double step) override {
    int d = Super::pFunc->dim();
    // m = beta1 * m + (1 - beta1) * g
    double beta1 = param.beta1;
    double beta1C = 1.0 - beta1;
    unary_op(_m.get(), pg, d, [beta1, beta1C](
        double& m, double const& g) { m = beta1 * m + beta1C * g; });
    // v = beta2 * v + (1 - beta2) * g .^ 2
    double beta2 = param.beta2;
    double beta2C = 1.0 - beta2;
    unary_op(_v.get(), pg, d, [beta2, beta2C](
        double& v, double const& g) { v = beta2 * v + beta2C * g * g; });
    // x = x - step * m / sqrt(v + eps)
    binary_op(px, _m.get(), _v.get(), d, [this, step](
        double& x, double const& m, double const& v) {
                x -= step * m / std::sqrt(v + this->param.eps); });
    Super::pFunc->update(px, d);
  }

  void fixedStepGD () override {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    if (!_m) { _m.reset(new double[d]); }
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_m.get(), d, 0.0);
    std::fill_n(_v.get(), d, 0.0);
    Super::fixedStepGD();
  }

  void adaptiveStepGD () override {
    if (param.maxIterations == 0) { return; }
    int d = Super::pFunc->dim();
    if (!_m) { _m.reset(new double[d]); }
    if (!_v) { _v.reset(new double[d]); }
    std::fill_n(_m.get(), d, 0.0);
    std::fill_n(_v.get(), d, 0.0);
    Super::adaptiveStepGD();
  }

  std::unique_ptr<double> _m;
  std::unique_ptr<double> _v;
};

};
};

#endif
