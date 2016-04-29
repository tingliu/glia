#ifndef _glia_alg_function_hxx_
#define _glia_alg_function_hxx_

#include "type/function.hxx"
#include "util/container.hxx"
#include "util/linalg.hxx"
#include "util/mp.hxx"

namespace glia {
namespace alg {

class Logsig : public virtual opt::TFunction<Eigen::VectorXd> {
 public:
  typedef opt::TFunction<Eigen::VectorXd> Super;
  typedef Logsig Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef Eigen::VectorXd Input;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (int D)
  { w = std::make_shared<Eigen::VectorXd>(D); }

  Logsig () {}

  Logsig (int D) { initialize(D); }

  ~Logsig () override {}

  virtual void scopy (Self const& x) { w = x.w; }

  double operator() (Input const& x) override
  { return 1.0 / (1.0 + std::exp(-(x.dot(*w)))); }

  double operator() (double* g, Input const& x) override {
    double fx = operator()(x);
    Eigen::Map<Eigen::VectorXd>(g, dim()) = fx * (1.0 - fx) * x;
    return fx;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override
  { if (w_ != w->data()) { std::copy_n(w_, d, w->data()); } }

  double* data () override { return w->data(); }

  double const* data () const override { return w->data(); }
};


class Norm2 : public opt::ConstFunction {
 public:
  typedef opt::ConstFunction Super;
  typedef Norm2 Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  std::shared_ptr<Eigen::VectorXd> w;

  virtual void initialize (std::shared_ptr<Eigen::VectorXd> w_) { w = w_; }

  Norm2 () {}

  Norm2 (std::shared_ptr<Eigen::VectorXd> w) { initialize(w); }

  ~Norm2 () override {}

  virtual void scopy (Self const& x) { w = x.w; }

  double operator() () override { return w->squaredNorm() / 2.0; }

  double operator() (double* g) override {
    std::copy_n(w->data(), w->size(), g);
    return w->squaredNorm() / 2.0;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override
  { if (w_ != w->data()) { std::copy_n(w_, d, w->data()); } }

  double* data () override { return w->data(); }

  double const* data () const override { return w->data(); }
};


// Quadratic loss function: |Y - F|^2 / 2
template <typename PFunc, typename TInput>
class TQuadraticLoss : public opt::TFunction<TInput> {
 public:
  typedef opt::TFunction<TInput> Super;
  typedef TQuadraticLoss<PFunc, TInput> Self;
  typedef PFunc PointFunction;
  typedef TInput Input;

  std::shared_ptr<PointFunction> f;
  // // Trial
  // int nNonZeroLoss;
  // // ~ Trial

  virtual void initialize (std::shared_ptr<PointFunction> f_) { f = f_; }

  TQuadraticLoss () {}

  TQuadraticLoss (std::shared_ptr<PointFunction> f) { initialize(f); }

  ~TQuadraticLoss () override {}

  virtual void scopy (Self const& x) {
    if (!f || f == x.f) { f = std::make_shared<PointFunction>(); }
    f->scopy(*x.f);
  }

  template <typename TPSamp, typename TPTar>
  double loss (
      std::vector<TPSamp> const& samples,
      std::vector<TPTar> const& targets)
  { return computeLoss(samples, targets, *f); }

  double operator() (Input const& x) override
  { return 0.5 * computeLoss(x.samples(), x.targets(), *f); }

  double operator() (double* g, Input const& x) override {
    Eigen::Map<Eigen::VectorXd> grad(g, dim());
    return 0.5 * computeLoss(grad, x.samples(), x.targets(), *f);
  }

  int dim () const override { return f->dim(); }

  void update (double const* w, int d) override { f->update(w, d); }

  double* data () override { return f->data(); }

  double const* data () const override { return f->data(); }

 protected:
  // Compute loss |Y - F|^2
  template <typename TPSamp, typename TPTar> double
  computeLoss (
      std::vector<TPSamp> const& samples,
      std::vector<TPTar> const& targets, PointFunction& f) {
    int n = samples.size();
    incvec(_m_loss, n, false);
    int nThreads = glia::nthreads();
    incvec(_m_fs, nThreads, true);
    for (int ti = 0; ti < nThreads; ++ti) { this->_m_fs[ti].scopy(f); }
    parfor(0, n, true, [this, &f, &samples, &targets](int i) {
        int ti = 0;
#ifdef GLIA_MT
        ti = omp_get_thread_num();
#endif
        this->_m_loss[i] = *targets[i] - this->_m_fs[ti](*samples[i]);
      }, 0);
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   _m_loss.resize(n);
    //   nNonZeroLoss = std::count_if(
    //       _m_loss.begin(), _m_loss.end(), [](double const& x) {
    //         return !isfeq(x, 0.0); });
    //   // printf("\tnNonZeroLoss = %-6d ", nNonZeroLoss);
    // }
    // // ~ Trial
    return Eigen::Map<Eigen::VectorXd>(_m_loss.data(), n).squaredNorm();
  }

  // Compute loss |Y - F|^2 and
  // loss gradient (d-by-1): (-gF)' * (Y - F)
  template <typename TPSamp, typename TPTar> double
  computeLoss (
      Eigen::Map<Eigen::VectorXd>& grad,
      std::vector<TPSamp> const& samples,
      std::vector<TPTar> const& targets, PointFunction& f) {
    int d = f.dim();
    int n = samples.size();
    grad.setZero();
    incvec(_m_loss, n, false);
    int nThreads = glia::nthreads();
    incvec(_m_fs, nThreads, true);
    incvec(_m_grads, nThreads, true);
    incvec(_m_gradSums, nThreads, true);
    for (int ti = 0; ti < nThreads; ++ti) { this->_m_fs[ti].scopy(f); }
    parfor(0, nThreads, false, [this, &f, d](int ti) {
        this->_m_grads[ti].resize(d);
        this->_m_gradSums[ti].setZero(d); }, 0);
    parfor(0, n, false, [this, &f, d, &samples, &targets](int i) {
        int ti = 0;
#ifdef GLIA_MT
        ti = omp_get_thread_num();
#endif
        this->_m_loss[i] = *targets[i] - this->_m_fs[ti](
            this->_m_grads[ti].data(), *samples[i]);
        if (!isfeq(this->_m_loss[i], 0.0))
        { _m_gradSums[ti] -= _m_grads[ti] * this->_m_loss[i]; }
      }, 0);
    for (Eigen::VectorXd const& g : _m_gradSums) { grad += g; }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   _m_loss.resize(n);
    //   nNonZeroLoss = std::count_if(
    //       _m_loss.begin(), _m_loss.end(), [](double const& x) {
    //         return !isfeq(x, 0.0); });
    //   // printf("\tnNonZeroLoss = %-6d ", nNonZeroLoss);
    // }
    // // ~ Trial
    return Eigen::Map<Eigen::VectorXd>(_m_loss.data(), n).squaredNorm();
  }

  friend Eigen::Map<Eigen::VectorXd>
  createEmptyEigenVectorMap (std::vector<double>& mem, int d);

  friend void
  incvec<PointFunction> (std::vector<PointFunction>& x, int n, bool keep);

  friend void
  incvec<double> (std::vector<double>& x, int n, bool keep);

  std::vector<double> _m_loss;
  std::vector<PointFunction> _m_fs;
  std::vector<Eigen::VectorXd> _m_grads;
  std::vector<Eigen::VectorXd> _m_gradSums;
};


// Log Gaussian with quadratic loss function:
//   |Y - F|^2 / 2 / sigma^2 + N * log(sigma)
template <typename PFunc, typename TInput>
class TLogGaussian : public TQuadraticLoss<PFunc, TInput> {
 public:
  typedef TQuadraticLoss<PFunc, TInput> Super;
  typedef TLogGaussian<PFunc, TInput> Self;
  typedef PFunc PointFunction;
  typedef TInput Input;

  double sigma2;  // Noise variance: sigma^2

  virtual void initialize (
      std::shared_ptr<PointFunction> f, double sigma) {
    Super::initialize(f);
    sigma2 = sigma * sigma;
  }

  TLogGaussian () {}

  TLogGaussian (std::shared_ptr<PointFunction> f, double sigma)
  { initialize(f, sigma); }

  ~TLogGaussian () override {}

  virtual void scopy (Self const& x) {
    Super::scopy(x);
    sigma2 = x.sigma2;
  }

  double operator() (Input const& x) override {
    return Super::operator()(x) / sigma2 +
        x.size() * std::log(sigma2) / 2.0;
  }

  double operator() (double* g, Input const& x) override {
    double ret = Super::operator()(g, x) / sigma2 +
        x.size() * std::log(sigma2) / 2.0;
    inplace_op(g, Super::dim(), [this](double& g) { g /= this->sigma2; });
    return ret;
  }
};


// Binary cross entropy loss function:
//   sum_i(-y_i * log(f_i) - (1 - y_i) * log(1 - f_i))
template <typename PFunc, typename TInput>
class TCrossEntropyLoss : public opt::TFunction<TInput> {
 public:
  typedef opt::TFunction<TInput> Super;
  typedef TCrossEntropyLoss<PFunc, TInput> Self;
  typedef PFunc PointFunction;
  typedef TInput Input;

  std::shared_ptr<PointFunction> f;
  // // Trial
  // int nNonZeroLoss;
  // // ~ Trial

  virtual void initialize (std::shared_ptr<PointFunction> f_) { f = f_; }

  TCrossEntropyLoss () {}

  TCrossEntropyLoss (std::shared_ptr<PointFunction> f) { initialize(f); }

  ~TCrossEntropyLoss () override {}

  virtual void scopy (Self const& x) {
    if (!f || f == x.f) { f = std::make_shared<PointFunction>(); }
    f->scopy(*x.f);
  }

  double operator() (Input const& x) override
  { return computeLoss(x.samples(), x.targets(), *f); }

  double operator() (double* g, Input const& x) override {
    Eigen::Map<Eigen::VectorXd> grad(g, dim());
    return computeLoss(grad, x.samples(), x.targets(), *f);
  }

  int dim () const override { return f->dim(); }

  void update (double const* w, int d) override { f->update(w, d); }

  double* data () override { return f->data(); }

  double const* data () const override { return f->data(); }

 protected:
  template <typename TPSamp, typename TPTar> double
  computeLoss (
      std::vector<TPSamp> const& samples,
      std::vector<TPTar> const& targets, PointFunction& f) {
    int n = samples.size();
    incvec(_m_loss, n, false);
    int nThreads = glia::nthreads();
    incvec(_m_fs, nThreads, true);
    for (int ti = 0; ti < nThreads; ++ti) { this->_m_fs[ti].scopy(f); }
    parfor(0, n, true, [this, &f, &samples, &targets](int i) {
        int ti = 0;
#ifdef GLIA_MT
        ti = omp_get_thread_num();
#endif
        double fx = this->_m_fs[ti](*samples[i]);
        this->_m_loss[i] = -std::log(
            std::max((*targets[i] > 0.5 ? fx : 1.0 - fx), FEPS));
      }, 0);
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   nNonZeroLoss = std::count_if(
    //       _m_loss.begin(), _m_loss.end(), [](double const& x) {
    //         return !isfeq(x, 0.0); });
    //   // printf("\tnNonZeroLoss = %-6d ", nNonZeroLoss);
    // }
    // // ~ Trial
    return Eigen::Map<Eigen::VectorXd>(_m_loss.data(), n).sum();
  }

  template <typename TPSamp, typename TPTar> double
  computeLoss (
      Eigen::Map<Eigen::VectorXd>& grad,
      std::vector<TPSamp> const& samples,
      std::vector<TPTar> const& targets, PointFunction& f) {
    int d = f.dim();
    int n = samples.size();
    grad.setZero();
    incvec(_m_loss, n, false);
    int nThreads = glia::nthreads();
    incvec(_m_fs, nThreads, true);
    incvec(_m_grads, nThreads, true);
    incvec(_m_gradSums, nThreads, true);
    for (int ti = 0; ti < nThreads; ++ti) { this->_m_fs[ti].scopy(f); }
    parfor(0, nThreads, false, [this, &f, d](int ti) {
        this->_m_grads[ti].resize(d);
        this->_m_gradSums[ti].setZero(d); }, 0);
    parfor(0, n, false, [this, &f, d, &samples, &targets](int i) {
        int ti = 0;
#ifdef GLIA_MT
        ti = omp_get_thread_num();
#endif
        double fx = this->_m_fs[ti](
            this->_m_grads[ti].data(), *samples[i]);
        if (*targets[i] > 0.5) {
          fx = std::max(fx, FEPS);
          this->_m_loss[i] = -std::log(fx);
          _m_gradSums[ti] -= _m_grads[ti] / fx;
        } else {
          fx = std::max(1.0 - fx, FEPS);
          this->_m_loss[i] = -std::log(fx);
          _m_gradSums[ti] += _m_grads[ti] / fx;
        }
      }, 0);
    for (Eigen::VectorXd const& g : _m_gradSums) { grad += g; }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   nNonZeroLoss = std::count_if(
    //       _m_loss.begin(), _m_loss.end(), [](double const& x) {
    //         return !isfeq(x, 0.0); });
    //   // printf("\tnNonZeroLoss = %-6d ", nNonZeroLoss);
    // }
    // // ~ Trial
    return Eigen::Map<Eigen::VectorXd>(_m_loss.data(), n).sum();
  }

  friend void
  incvec<PointFunction> (std::vector<PointFunction>& x, int n, bool keep);

  friend void
  incvec<double> (std::vector<double>& x, int n, bool keep);

  std::vector<double> _m_loss;
  std::vector<PointFunction> _m_fs;
  std::vector<Eigen::VectorXd> _m_grads;
  std::vector<Eigen::VectorXd> _m_gradSums;
};

};
};

#endif
