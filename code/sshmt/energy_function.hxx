#ifndef _glia_sshmt_energy_function_hxx_
#define _glia_sshmt_energy_function_hxx_

#include "type/energy_function.hxx"
#include "alg/function.hxx"
#include "alg/dnf.hxx"

namespace glia {
namespace sshmt {

template <typename CFunc, typename TInput>
class MonotonicDnfGaussianCrossEntropy
    : public opt::TRegularizedEnergyFunction<
  alg::Norm2,
  alg::TLogGaussian<
    alg::MonotonicDNF<CFunc>, typename TInput::Unsupervised>,
  alg::TCrossEntropyLoss<CFunc, typename TInput::Supervised>, TInput> {
public:
  typedef opt::TRegularizedEnergyFunction<
   alg::Norm2,
   alg::TLogGaussian<
     alg::MonotonicDNF<CFunc>, typename TInput::Unsupervised>,
   alg::TCrossEntropyLoss<
     CFunc, typename TInput::Supervised>, TInput> Super;
  typedef MonotonicDnfGaussianCrossEntropy Self;
  typedef TInput Input;
  typedef typename Super::Regularizer Regularizer;
  typedef typename Super::Unsupervised Unsupervised;
  typedef typename Super::Supervised Supervised;
  typedef alg::MonotonicDNF<CFunc> Dnf;
  typedef CFunc Classifier;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU) {
    Super::initialize(wr, wu, ws);
    w = f->w;
    if (Super::regularizer) { Super::regularizer->initialize(w); }
    if (Super::unsupervised) {
      Super::unsupervised->initialize(
          std::make_shared<Dnf>(f, mergeTarget), sigmaU);
    }
    if (Super::supervised) { Super::supervised->initialize(f); }
  }

  MonotonicDnfGaussianCrossEntropy () {}

  MonotonicDnfGaussianCrossEntropy (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU)
  { initialize(f, wr, wu, ws, mergeTarget, sigmaU); }

  ~MonotonicDnfGaussianCrossEntropy () override {}

  virtual void scopy (Self& x) {
    Super::scopy(x);
    w = x.w;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override {
    if (w_ != w->data()) { std::copy_n(w_, d, w->data()); }
    Super::update(w->data(), d);
  }

  // Copy itself to w_
  virtual void copy (double* w_) const
  { std::copy_n(w->data(), w->size(), w_); }
};


// Regularized MAP energy function using monotonic DNF
template <typename CFunc, typename TInput>
class MonotonicDnfGaussian : public opt::TRegularizedEnergyFunction<
  alg::Norm2,
  alg::TLogGaussian<
    alg::MonotonicDNF<CFunc>, typename TInput::Unsupervised>,
  alg::TLogGaussian<CFunc, typename TInput::Supervised>, TInput> {
 public:
  typedef opt::TRegularizedEnergyFunction<
   alg::Norm2,
   alg::TLogGaussian<
     alg::MonotonicDNF<CFunc>, typename TInput::Unsupervised>,
   alg::TLogGaussian<
     CFunc, typename TInput::Supervised>, TInput> Super;
  typedef MonotonicDnfGaussian Self;
  typedef TInput Input;
  typedef typename Super::Regularizer Regularizer;
  typedef typename Super::Unsupervised Unsupervised;
  typedef typename Super::Supervised Supervised;
  typedef alg::MonotonicDNF<CFunc> Dnf;
  typedef CFunc Classifier;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU, double sigmaS) {
    Super::initialize(wr, wu, ws);
    w = f->w;
    if (Super::regularizer) { Super::regularizer->initialize(w); }
    if (Super::unsupervised) {
      Super::unsupervised->initialize(
          std::make_shared<Dnf>(f, mergeTarget), sigmaU);
    }
    if (Super::supervised)
    { Super::supervised->initialize(f, sigmaS); }
  }

  MonotonicDnfGaussian () {}

  MonotonicDnfGaussian (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU, double sigmaS)
  { initialize(f, wr, wu, ws, mergeTarget, sigmaU, sigmaS); }

  ~MonotonicDnfGaussian () override {}

  virtual void scopy (Self& x) {
    Super::scopy(x);
    w = x.w;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override {
    if (w_ != w->data()) { std::copy_n(w_, d, w->data()); }
    Super::update(w->data(), d);
  }

  // Copy itself to w_
  virtual void copy (double* w_) const
  { std::copy_n(w->data(), w->size(), w_); }
};


// Regularized MAP energy function using monotonic DNF
template <typename CFunc, typename TInput>
class RelaxedMonotonicDnfGaussian
    : public opt::TRegularizedEnergyFunction<
  alg::Norm2,
  alg::TLogGaussian<
    alg::RelaxedMonotonicDNF<CFunc>, typename TInput::Unsupervised>,
  alg::TLogGaussian<CFunc, typename TInput::Supervised>, TInput> {
 public:
  typedef opt::TRegularizedEnergyFunction<
   alg::Norm2,
   alg::TLogGaussian<
     alg::RelaxedMonotonicDNF<CFunc>, typename TInput::Unsupervised>,
   alg::TLogGaussian<
     CFunc, typename TInput::Supervised>, TInput> Super;
  typedef RelaxedMonotonicDnfGaussian Self;
  typedef TInput Input;
  typedef typename Super::Regularizer Regularizer;
  typedef typename Super::Unsupervised Unsupervised;
  typedef typename Super::Supervised Supervised;
  typedef alg::RelaxedMonotonicDNF<CFunc> Dnf;
  typedef CFunc Classifier;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double sigmaU, double sigmaS) {
    Super::initialize(wr, wu, ws);
    w = f->w;
    if (Super::regularizer) { Super::regularizer->initialize(w); }
    if (Super::unsupervised) {
      Super::unsupervised->initialize(std::make_shared<Dnf>(f), sigmaU);
    }
    if (Super::supervised)
    { Super::supervised->initialize(f, sigmaS); }
  }

  RelaxedMonotonicDnfGaussian () {}

  RelaxedMonotonicDnfGaussian (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double sigmaU, double sigmaS)
  { initialize(f, wr, wu, ws, sigmaU, sigmaS); }

  ~RelaxedMonotonicDnfGaussian () override {}

  virtual void scopy (Self& x) {
    Super::scopy(x);
    w = x.w;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override {
    if (w_ != w->data()) { std::copy_n(w_, d, w->data()); }
    Super::update(w->data(), d);
  }

  // Copy itself to w_
  virtual void copy (double* w_) const
  { std::copy_n(w->data(), w->size(), w_); }
};


// Regularized MAP energy function using unique DNF
template <typename CFunc, typename TInput>
class UniqueDnfGaussian : public opt::TRegularizedEnergyFunction<
  alg::Norm2,
  alg::TLogGaussian<
    alg::UniqueDNF<CFunc>, typename TInput::Unsupervised>,
  alg::TLogGaussian<CFunc, typename TInput::Supervised>, TInput> {
 public:
  typedef opt::TRegularizedEnergyFunction<
   alg::Norm2,
   alg::TLogGaussian<
     alg::UniqueDNF<CFunc>, typename TInput::Unsupervised>,
   alg::TLogGaussian<
     CFunc, typename TInput::Supervised>, TInput> Super;
  typedef UniqueDnfGaussian Self;
  typedef TInput Input;
  typedef typename Super::Regularizer Regularizer;
  typedef typename Super::Unsupervised Unsupervised;
  typedef typename Super::Supervised Supervised;
  typedef alg::UniqueDNF<CFunc> Dnf;
  typedef CFunc Classifier;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU, double sigmaS) {
    Super::initialize(wr, wu, ws);
    w = f->w;
    if (Super::regularizer) { Super::regularizer->initialize(w); }
    if (Super::unsupervised) {
      Super::unsupervised->initialize(
          std::make_shared<Dnf>(f, mergeTarget), sigmaU);
    }
    if (Super::supervised)
    { Super::supervised->initialize(f, sigmaS); }
  }

  UniqueDnfGaussian () {}

  UniqueDnfGaussian (
      std::shared_ptr<Classifier> f, double wr, double wu, double ws,
      double mergeTarget, double sigmaU, double sigmaS)
  { initialize(f, wr, wu, ws, mergeTarget, sigmaU, sigmaS); }

  ~UniqueDnfGaussian () override {}

  virtual void scopy (Self& x) {
    Super::scopy(x);
    w = x.w;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override {
    if (w_ != w->data()) { std::copy_n(w_, d, w->data()); }
    Super::update(w->data(), d);
  }

  // Copy itself to w_
  virtual void copy (double* w_) const
  { std::copy_n(w->data(), w->size(), w_); }
};

};
};

#endif
