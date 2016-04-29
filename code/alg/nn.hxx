#ifndef _glia_alg_nn_hxx_
#define _glia_alg_nn_hxx_

#include "type/function.hxx"
#include "util/text_io.hxx"
#include "Eigen/Dense"

namespace glia {
namespace alg {

// Multilayer perceptron of two hidden layers with
// * Relu activation functions for hidden layers
// * Logsig activation function for output layer
class MLP2 : public virtual opt::TFunction<Eigen::VectorXd> {
 public:
  typedef opt::TFunction<Eigen::VectorXd> Super;
  typedef MLP2 Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef Eigen::VectorXd Input;

  std::shared_ptr<Eigen::VectorXd> w;
  int D;
  int N1;
  int N2;

  // if isWholeD, D_ is total weight dimension
  virtual void initialize (bool isWholeD, int D_, int N1_, int N2_) {
    N1 = N1_;
    N2 = N2_;
    D = isWholeD ? (D_ - (N1 + 1) * N2 - N2 - 1) / N1 : D_;
    w = std::make_shared<Eigen::VectorXd>();
    w->resize(D * N1 + (N1 + 1) * N2 + N2 + 1);
    setUpPtrs();
  }

  MLP2 () {}

  MLP2 (bool isWholeD, int D, int N1, int N2)
  { initialize(isWholeD, D, N1, N2); }

  ~MLP2 () override {}

  virtual void scopy (Self const& x) {
    w = x.w;
    D = x.D;
    N1 = x.N1;
    N2 = x.N2;
    setUpPtrs();
  }

  double operator() (Input const& x) override {
    Eigen::Map<Eigen::MatrixXd> w0(_m_w0, D, N1);
    Eigen::Map<Eigen::MatrixXd> w1(_m_w1, N1 + 1, N2);
    Eigen::Map<Eigen::VectorXd> w2(_m_w2, N2 + 1);
    _m_h1.resize(N1);
    _m_h1a.resize(N1 + 1);
    _m_h2.resize(N2);
    _m_h2a.resize(N2 + 1);
    _m_h1 = x.transpose() * w0;
    _m_h1a.leftCols(N1) = _m_h1.unaryExpr(
        [](double hx) { return hx > 0.0 ? hx : 0.0; });
    _m_h1a.rightCols(1).setOnes();
    _m_h2 = _m_h1a * w1;
    _m_h2a.leftCols(N2) = _m_h2.unaryExpr(
        [](double hx) { return hx > 0.0 ? hx : 0.0; });
    _m_h2a.rightCols(1).setOnes();
    double h3 = _m_h2a * w2;
    double f = 1.0 / (1.0 + std::exp(-h3));
    return f;
  }

  double operator() (double* g, Input const& x) override {
    Eigen::Map<Eigen::MatrixXd> w0(_m_w0, D, N1);
    Eigen::Map<Eigen::MatrixXd> w1(_m_w1, N1 + 1, N2);
    Eigen::Map<Eigen::VectorXd> w2(_m_w2, N2 + 1);
    Eigen::Map<Eigen::MatrixXd> dw0(g, D, N1);
    Eigen::Map<Eigen::MatrixXd> dw1(g + D * N1, N1 + 1, N2);
    Eigen::Map<Eigen::VectorXd> dw2(g + D * N1 + (N1 + 1) * N2, N2 + 1);
    _m_h1.resize(N1);
    _m_h1a.resize(N1 + 1);
    _m_h2.resize(N2);
    _m_h2a.resize(N2 + 1);
    _m_dh1.resize(N1 + 1);
    _m_dh2.resize(N2 + 1);
    _m_h1 = x.transpose() * w0;
    _m_h1a.leftCols(N1) = _m_h1.unaryExpr(
        [](double hx) { return hx > 0.0 ? hx : 0.0; });
    _m_h1a.rightCols(1).setOnes();
    _m_h2 = _m_h1a * w1;
    _m_h2a.leftCols(N2) = _m_h2.unaryExpr(
        [](double hx) { return hx > 0.0 ? hx : 0.0; });
    _m_h2a.rightCols(1).setOnes();
    double h3 = _m_h2a * w2;
    double f = 1.0 / (1.0 + std::exp(-h3));
    double dh3 = f * (1.0 - f);
    dw2 = _m_h2a.transpose() * dh3;
    _m_dh2 = dh3 * w2.transpose();
    for (int i = 0; i < N2; ++i)
    { if (_m_h2(i) <= 0.0) { _m_dh2(i) = 0.0; } }
    dw1 = _m_h1a.transpose() * _m_dh2.leftCols(N2);
    _m_dh1 = _m_dh2.leftCols(N2) * w1.transpose();
    for (int i = 0; i < N1; ++i)
    { if (_m_h1(i) <= 0.0) { _m_dh1(i) = 0.0; } }
    dw0 = x * _m_dh1.leftCols(N1);
    return f;
  }

  int dim () const override { return w->size(); }

  void update (double const* w_, int d) override
  { if (w_ != w->data()) { std::copy_n(w_, d, w->data()); } }

  double* data () override { return w->data(); }

  double const* data () const override { return w->data(); }

 protected:
  void setUpPtrs () {
    _m_w0 = w->data();
    _m_w1 = _m_w0 + D * N1;
    _m_w2 = _m_w1 + (N1 + 1) * N2;
  }

  double* _m_w0;  // D * N1
  double* _m_w1;  // (N1 + 1) * N2
  double* _m_w2;  // (N2 + 1) * 1
  Eigen::RowVectorXd _m_h1;  // 1 * N1
  Eigen::RowVectorXd _m_h1a;  // 1 * (N1 + 1)
  Eigen::RowVectorXd _m_h2;  // 1 * N2
  Eigen::RowVectorXd _m_h2a;  // 1 * (N2 + 1)
  Eigen::RowVectorXd _m_dh1;  // 1 * (N1 + 1)
  Eigen::RowVectorXd _m_dh2;  // 1 * (N2 + 1)
};


class MLP2v : public virtual opt::TFunction<std::vector<FVal>> {
 public:
  typedef opt::TFunction<std::vector<FVal>> Super;
  typedef MLP2v Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef std::vector<FVal> Input;

  virtual void initialize (bool isWholeD, int D_, int N1_, int N2_) {
    if (!_mlp) { _mlp = std::make_shared<MLP2>(isWholeD, D_, N1_, N2_); }
    else { _mlp->initialize(isWholeD, D_, N1_, N2_); }
  }

  virtual void initialize (int N1, int N2, std::string const& modelFile) {
    std::vector<double> w;
    readData(w, modelFile, true);
    initialize(true, w.size(), N1, N2);
  }

  MLP2v () {}

  MLP2v (bool isWholeD, int D, int N1, int N2)
  { initialize(isWholeD, D, N1, N2); }

  MLP2v (int N1, int N2, std::string const& modelFile)
  { initialize(N1, N2, modelFile); }

  ~MLP2v () override {}

  virtual void scopy (Self const& x) { _mlp->scopy(*x._mlp); }

  double operator() (Input const& x) override {
    return _mlp->operator()(
        Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()));
  }

  double operator() (double* g, Input const& x) override {
    return _mlp->operator()(
        g, Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()));
  }

  int dim () const override { return _mlp->dim(); }

  void update (double const* w_, int d) override { _mlp->update(w_, d); }

  double* data () override { return _mlp->data(); }

  double const* data () const override { return _mlp->data(); }

 protected:
  std::shared_ptr<MLP2> _mlp;
};


class EnsembleMLP2v : public virtual opt::TFunction<std::vector<FVal>> {
 public:
  typedef opt::TFunction<std::vector<FVal>> Super;
  typedef EnsembleMLP2v Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef std::vector<FVal> Input;
  typedef std::function<int(Input)> Distributor;

  std::vector<std::shared_ptr<MLP2>> models;
  Distributor fdist;

  virtual void initialize (
      int N1, int N2, std::vector<std::string> const& modelFiles,
      Distributor const& fdist_) {
    int n = modelFiles.size();
    models.resize(n);
    std::vector<double> w;
    for (int i = 0; i < n; ++i) {
      w.clear();
      readData(w, modelFiles[i], true);
      models[i] = std::make_shared<MLP2>(true, w.size(), N1, N2);
      models[i]->update(w.data(), w.size());
    }
    fdist = fdist_;
  }

  EnsembleMLP2v () {}

  EnsembleMLP2v (
      int N1, int N2, std::vector<std::string> const& modelFiles,
      Distributor const& fdist)
  { initialize(N1, N2, modelFiles, fdist); }

  ~EnsembleMLP2v () override {}

  double operator() (Input const& x) override {
    return models[fdist(x)]->operator()(
        Eigen::Map<const Eigen::VectorXd>(x.data(), x.size()));
  }

  double operator() (double* g, Input const& x) override {
    perr("Error: no gradient available for ensemble MLP2...");
    return operator()(x);
  }

  int dim () const override {
    perr("Error: no dimension available for ensemble MLP2...");
    return -1;
  }

  void update (double const* w_, int d) override
  { perr("Error: no update can be done to ensemble MLP2..."); }

  double* data () override {
    perr("Error: no data pointer can be returned for ensemble MLP2...");
    return nullptr;
  }

  double const* data () const override {
    perr("Error: no data pointer can be returned for ensemble MLP2...");
    return nullptr;
  }
};

};
};

#endif
