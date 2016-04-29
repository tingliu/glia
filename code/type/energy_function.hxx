#ifndef _glia_type_energy_function_hxx_
#define _glia_type_energy_function_hxx_

#include "type/function.hxx"
#include "type/function_input.hxx"
#include "util/container.hxx"

namespace glia {
namespace opt {

template <typename TUns, typename TSu>
class TEnergyFunctionInput : public FunctionInput {
 public:
  typedef FunctionInput Super;
  typedef TEnergyFunctionInput<TUns, TSu> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TUns Unsupervised;
  typedef TSu Supervised;

  bool useUnsupervised = true;
  bool useSupervised = true;
  std::shared_ptr<Unsupervised> unsupervised;
  std::shared_ptr<Supervised> supervised;

  ~TEnergyFunctionInput () override {}

  int size () const override {
    int ret = 0;
    if (useUnsupervised) { ret += unsupervised->size(); }
    if (useSupervised) { ret += supervised->size(); }
    return ret;
  }

  int dim () const override {
    int du = 0, ds = 0;
    if (useUnsupervised) { du = unsupervised->dim(); }
    if (useSupervised) { ds = supervised->dim(); }
    return std::max(du, ds);
  }

  BatchSampler::BatchType prepareBatch () override {
    BatchSampler::BatchType ret = BatchSampler::BatchType::None;
    if (useUnsupervised)
    { ret = std::max(ret, unsupervised->prepareBatch()); }
    if (useSupervised)
    { ret = std::max(ret, supervised->prepareBatch()); }
    return ret;
  }

  bool isUsingBatchSampler () const override {
    return (useUnsupervised && unsupervised->isUsingBatchSampler()) ||
        (useSupervised && supervised->isUsingBatchSampler());
  }

  void turnOnUseAll () override {
    if (useUnsupervised) { unsupervised->turnOnUseAll(); }
    if (useSupervised) { supervised->turnOnUseAll(); }
  }

  void turnOffUseAll () override {
    if (useUnsupervised) { unsupervised->turnOffUseAll(); }
    if (useSupervised) { supervised->turnOffUseAll(); }
  }
};


template <typename RFunc, typename UFunc, typename SFunc, typename TInput>
class TRegularizedEnergyFunction : public TFunction<TInput> {
 public:
  typedef TFunction<TInput> Super;
  typedef TRegularizedEnergyFunction<RFunc, UFunc, SFunc, TInput> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef RFunc Regularizer;
  typedef UFunc Unsupervised;
  typedef SFunc Supervised;
  typedef TInput Input;

  bool useRegularizer = true;
  bool useUnsupervised = true;
  bool useSupervised = true;
  double wr;
  double wu;
  double ws;
  std::shared_ptr<Regularizer> regularizer;
  std::shared_ptr<Unsupervised> unsupervised;
  std::shared_ptr<Supervised> supervised;

  virtual void initialize (double wr_, double wu_, double ws_) {
    wr = wr_;
    wu = wu_;
    ws = ws_;
    regularizer.reset(new Regularizer);
    unsupervised.reset(new Unsupervised);
    supervised.reset(new Supervised);
    useRegularizer = !isfeq(wr, 0.0);
    useSupervised = !isfeq(ws, 0.0);
    useUnsupervised = !isfeq(wu, 0.0);
  }

  TRegularizedEnergyFunction () {}

  TRegularizedEnergyFunction (double wr, double wu, double ws)
  { initialize(wr, wu, ws); }

  ~TRegularizedEnergyFunction () override {}

  virtual void scopy (Self const& x) {
    wr = x.wr;
    wu = x.wu;
    ws = x.ws;
    if (!regularizer || regularizer == x.regularizer)
    { regularizer = std::make_shared<Regularizer>(); }
    if (!unsupervised || unsupervised == x.unsupervised)
    { unsupervised = std::make_shared<Unsupervised>(); }
    if (!supervised || supervised == x.supervised)
    { supervised = std::make_shared<Supervised>(); }
    regularizer->scopy(*x.regularizer);
    unsupervised->scopy(*x.unsupervised);
    supervised->scopy(*x.supervised);
  }

  int dim () const override {
    int dr = 0, du = 0, ds = 0;
    if (regularizer) { dr = regularizer->dim(); }
    if (unsupervised) { du = unsupervised->dim(); }
    if (supervised) { ds = supervised->dim(); }
    return std::max(dr, std::max(du, ds));
  }

  void update (double const* w, int d) override {
    if (regularizer) { regularizer->update(w, d); }
    if (unsupervised) { unsupervised->update(w, d); }
    if (supervised) { supervised->update(w, d); }
  }

  double* data () override {
    double* r = regularizer ? regularizer->data() : nullptr;
    double* u = unsupervised ? unsupervised->data() : nullptr;
    double* s = supervised ? supervised->data() : nullptr;
    if (r == u && u == s) { return r; }
    if (r == nullptr && u == s) { return u; }
    if (u == nullptr && r == s) { return r; }
    if (s == nullptr && r == u) { return r; }
    perr("Error: data() cannot be called, "
         "because no such data pointer...");
    return nullptr;
  }

  double const* data () const override {
    double* r = regularizer ? regularizer->data() : nullptr;
    double* u = unsupervised ? unsupervised->data() : nullptr;
    double* s = supervised ? supervised->data() : nullptr;
    if (r == u && u == s) { return r; }
    if (r == nullptr && u == s) { return u; }
    if (u == nullptr && r == s) { return r; }
    if (s == nullptr && r == u) { return r; }
    perr("Error: data() cannot be called, "
         "because no such data pointer...");
    return nullptr;
  }

  double operator() (Input const& x) override {
    double ret = 0.0;
    if (useRegularizer) { ret += wr * regularizer->operator()(); }
    if (useUnsupervised)
    { ret += wu * unsupervised->operator()(*x.unsupervised); }
    if (useSupervised)
    { ret += ws * supervised->operator()(*x.supervised); }
    return ret;
  }

  double operator() (double* g, Input const& x) override {
    int d = dim();
    incvec(_m_tmp_grad, d, false);
    double ret = 0.0;
    std::fill_n(g, d, 0.0);
    if (useRegularizer) {
      ret += wr * regularizer->operator()(_m_tmp_grad.data());
      if (isfeq(wr, 1.0)) {
        unary_op(g, _m_tmp_grad.data(), d, [](
            double& g, double const& t) { g += t; });
      } else {
        unary_op(g, _m_tmp_grad.data(), d, [this](
            double& g, double const& t) { g += this->wr * t; });
      }
      // // Debug
      // std::cout << "|gr| = " << std::sqrt(
      //     opSquaredNorm(_m_tmp_grad.data(), d)) << ", ";
      // // ~ Debug
    }
    if (useUnsupervised) {
      // // Trial
      // DO_WEIRD_STUFF = 1;
      // // ~ Trial
      ret += wu * unsupervised->operator()(
          _m_tmp_grad.data(), *x.unsupervised);
      // // Trial
      // DO_WEIRD_STUFF = 0;
      // // ~ Trial
      if (isfeq(wu, 1.0)) {
        unary_op(g, _m_tmp_grad.data(), d, [](
            double& g, double const& t) { g += t; });
      } else {
        unary_op(g, _m_tmp_grad.data(), d, [this](
            double& g, double const& t) { g += this->wu * t; });
      }
      // // Debug
      // std::cout << "|gu| = " << std::sqrt(
      //     opSquaredNorm(_m_tmp_grad.data(), d)) << ", ";
      // // ~ Debug
    }
    if (useSupervised) {
      ret += ws * supervised->operator()(
          _m_tmp_grad.data(), *x.supervised);
      if (isfeq(ws, 1.0)) {
        unary_op(g, _m_tmp_grad.data(), d, [](
            double& g, double const& t) { g += t; });
      } else {
        unary_op(g, _m_tmp_grad.data(), d, [this](
            double& g, double const& t) { g += this->ws * t; });
      }
      // // Debug
      // std::cout << "|gs| = " << std::sqrt(
      //     opSquaredNorm(_m_tmp_grad.data(), d)) << std::endl;
      // // ~ Debug
    }
    return ret;
  }

 protected:
  friend void
  incvec<double>(std::vector<double>& x, int n, bool keep);

  std::vector<double> _m_tmp_grad;
};

};
};

#endif
