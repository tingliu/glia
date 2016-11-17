#ifndef _glia_alg_dnf_hxx_
#define _glia_alg_dnf_hxx_

#include "util/mp.hxx"
#include "util/linalg.hxx"

namespace glia {
namespace alg {

template <typename CFunc>
class RelaxedMonotonicDNF : public opt::TFunction<Eigen::MatrixXd> {
 public:
  typedef opt::TFunction<Eigen::MatrixXd> Super;
  typedef RelaxedMonotonicDNF<CFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef CFunc Classifier;

  // Classifier function must be a batch function
  std::shared_ptr<Classifier> f;

  virtual void initialize (std::shared_ptr<Classifier> f_) { f = f_; }

  RelaxedMonotonicDNF () {}

  RelaxedMonotonicDNF (std::shared_ptr<Classifier> f) { initialize(f); }

  ~RelaxedMonotonicDNF () override {}

  virtual void scopy (Self const& x) {
    if (!f || f == x.f) { f = std::make_shared<Classifier>(); }
    f->scopy(*x.f);
  }

  double operator() (Eigen::MatrixXd const& x) override
  { return helperRelaxedMonotonicDNF(x, *f); }

  double operator() (double* g, Eigen::MatrixXd const& x) override
  { return helperRelaxedMonotonicDNF(g, x, *f); }

  int dim () const override { return f->dim(); }

  void update (double const* w, int d) override { f->update(w, d); }

  double* data () override { return f->data(); }

  double const* data () const override { return f->data(); }

 protected:
  template <typename Func> double
  helperRelaxedMonotonicDNF (Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    if (n == 0) { return 0.0; }
    _m_fxMat.resize(n, n + 1);
    for (int i = 0; i < n; ++i) { _m_fxMat.rightCols(1)(i) = f(x.col(i)); }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   bool isConsistent = true;
    //   for (int i = 1; i < n; ++i) {
    //     // Inconsistent if (f0 == 0 && f1 == 1)
    //     if (_m_fxMat.rightCols(1)(i - 1) < 0.5 &&
    //         _m_fxMat.rightCols(1)(i) > 0.5) {
    //       isConsistent = false;
    //       break;
    //     }
    //   }
    //   if (isConsistent) { return 1.0; }
    // }
    // // ~ Trial
    for (int i = n - 1; i >= 0; --i) {
      _m_fxMat.col(i) = _m_fxMat.col(i + 1);
      _m_fxMat(i, i) = 1.0 - _m_fxMat(i, i);
    }
    return _m_fxMat.colwise().prod().sum();
  }

  template <typename Func> double
  helperRelaxedMonotonicDNF (
      double* g, Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    if (n == 0) { return 0.0; }
    int d = f.dim();
    _m_fxMat.resize(n, n + 1);
    _m_gxMat.resize(d, n);
    _m_efxMat.resize(n - 1, n + 1);
    _m_efxColProds.resize(n + 1);
    for (int i = 0; i < n; ++i)
    { _m_fxMat.rightCols(1)(i) = f(_m_gxMat.col(i).data(), x.col(i)); }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   bool isConsistent = true;
    //   for (int i = 1; i < n; ++i) {
    //     // Inconsistent if (f0 == 0 && f1 == 1)
    //     if (_m_fxMat.rightCols(1)(i - 1) < 0.5 &&
    //         _m_fxMat.rightCols(1)(i) > 0.5) {
    //       isConsistent = false;
    //       break;
    //     }
    //   }
    //   if (isConsistent) { return 1.0; }
    // }
    // // ~ Trial
    for (int i = n - 1; i >= 0; --i) {
      _m_fxMat.col(i) = _m_fxMat.col(i + 1);
      _m_fxMat(i, i) = 1.0 - _m_fxMat(i, i);
    }
    Eigen::Map<Eigen::VectorXd> grad(g, d);
    grad.setZero();
    for (int i = 0; i < n; ++i) {
      _m_efxMat.topRows(i) = _m_fxMat.topRows(i);
      _m_efxMat.bottomRows(n - 1 - i) = _m_fxMat.bottomRows(n - 1 - i);
      _m_efxColProds = _m_efxMat.colwise().prod();
      _m_efxColProds.head(i + 1) = _m_efxColProds.head(i + 1);
      grad += _m_gxMat.col(i) * _m_efxColProds.prod();
    }
    return _m_fxMat.colwise().prod().sum();
  }

  Eigen::MatrixXd _m_fxMat;
  Eigen::MatrixXd _m_gxMat;
  Eigen::MatrixXd _m_efxMat;
  Eigen::VectorXd _m_efxColProds;
};


template <typename CFunc>
class MonotonicDNF : public opt::TFunction<Eigen::MatrixXd> {
 public:
  typedef opt::TFunction<Eigen::MatrixXd> Super;
  typedef MonotonicDNF<CFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef CFunc Classifier;

  // Classifier function must be a batch function
  std::shared_ptr<Classifier> f;
  double target;

  virtual void initialize (
      std::shared_ptr<Classifier> f_, double target_) {
    f = f_;
    target = target_;
  }

  MonotonicDNF () {}

  MonotonicDNF (std::shared_ptr<Classifier> f, double target)
  { initialize(f, target); }

  ~MonotonicDNF () override {}

  virtual void scopy (Self const& x) {
    if (!f || f == x.f) { f = std::make_shared<Classifier>(); }
    f->scopy(*x.f);
    target = x.target;
  }

  double operator() (Eigen::MatrixXd const& x) override
  { return helperMonotonicDNF(x, *f); }

  double operator() (double* g, Eigen::MatrixXd const& x) override
  { return helperMonotonicDNF(g, x, *f); }

  int dim () const override { return f->dim(); }

  void update (double const* w, int d) override { f->update(w, d); }

  double* data () override { return f->data(); }

  double const* data () const override { return f->data(); }

 protected:
  // fxMat: n-by-(n + 1) matrix, last column should be initialized
  //   Example: n = 3
  //     Input:      fxMat = [  -   -   -   f1
  //                            -   -   -   f2
  //                            -   -   -   f3 ]
  //     Output:     fxMat = [ ~f1  f1  f1  f1
  //                           ~f2 ~f2  f2  f2
  //                           ~f3 ~f3 ~f3  f3 ]
  //             cColProds = [ tar^3 - ~f1 * ~f2 * ~f3
  //                           tar^3 -  f1 * ~f2 * ~f3
  //                           tar^3 -  f1 *  f2 * ~f3
  //                           tar^3 -  f1 *  f2 *  f3 ]
  // return: 1 - (tar^3 - ~f1 * ~f2 * ~f3) * (tar^3 - f1 * ~f2 * ~f3) *
  //      (tar^3 - f1 * f2 * ~f3) * (tar^3 - f1 * f2 * f3)
  // return: 1 - exp(log(tar^3 - exp(log(~f1) + log(~f2) + log(~f3))) +
  //                 log(tar^3 - exp(log(f1) + log(~f2) + log(~f3))) +
  //                 log(tar^3 - exp(log(f1) + log(f2) + log(~f3))) +
  //                 log(tar^3 - exp(log(f1) + log(f2) + log(f3))))
  double helperMonotonicDNF (
      Eigen::Map<Eigen::VectorXd>& cColProds,
      Eigen::Map<Eigen::MatrixXd>& fxMat) {
    int n = fxMat.rows();
    for (int i = n - 1; i >= 0; --i) {
      fxMat.col(i) = fxMat.col(i + 1);
      fxMat(i, i) = 1.0 - fxMat(i, i);
    }
    cColProds = std::pow(target, n) - fxMat.colwise().prod().array();
    return 1.0 - cColProds.prod();
  }

  // x: d-by-n sample matrix
  // f: function applied to x; should return n-by-1 vector
  template <typename Func> double
  helperMonotonicDNF (Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    if (n == 0) { return 0.0; }
    Eigen::Map<Eigen::MatrixXd> fxMat =
        createEmptyEigenMatrixMap(_m_fxMat, n, n + 1);
    for (int i = 0; i < n; ++i) { fxMat.rightCols<1>()(i) = f(x.col(i)); }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   bool isConsistent = true;
    //   for (int i = 1; i < n; ++i) {
    //     // Inconsistent if (f0 == 0 && f1 == 1)
    //     if (fxMat.rightCols(1)(i - 1) < 0.5 &&
    //         fxMat.rightCols(1)(i) > 0.5) {
    //       isConsistent = false;
    //       break;
    //     }
    //   }
    //   if (isConsistent) { return 1.0; }
    // }
    // // ~ Trial
    Eigen::Map<Eigen::VectorXd> cColProds =
        createEmptyEigenVectorMap(_m_cColProds, n + 1);
    return helperMonotonicDNF(cColProds, fxMat);
  }

  // x: dx-by-n sample matrix
  // f: gradient function applied to x and fxMat
  // g: d-by-n gradient matrix
  template <typename Func> double
  helperMonotonicDNF (double* g, Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    int d = f.dim();
    // fxMat = [ ~f1  f1  f1  f1
    //           ~f2 ~f2  f2  f2
    //           ~f3 ~f3 ~f3  f3 ]
    Eigen::Map<Eigen::MatrixXd> fxMat =
        createEmptyEigenMatrixMap(_m_fxMat, n, n + 1);
    // gxMat = [ f1' f2' f3' ]
    Eigen::Map<Eigen::MatrixXd> gxMat =
        createEmptyEigenMatrixMap(_m_gxMat, d, n);
    for (int i = 0; i < n; ++i)
    { fxMat.rightCols<1>()(i) = f(gxMat.col(i).data(), x.col(i)); }
    // // Trial
    // if (DO_WEIRD_STUFF) {
    //   bool isConsistent = true;
    //   for (int i = 1; i < n; ++i) {
    //     // Inconsistent if (f0 == 0 && f1 == 1)
    //     if (fxMat.rightCols(1)(i - 1) < 0.5 &&
    //         fxMat.rightCols(1)(i) > 0.5) {
    //       isConsistent = false;
    //       break;
    //     }
    //   }
    //   if (isConsistent) { return 1.0; }
    // }
    // // ~ Trial
    // Complement column products: (n + 1)-by-1
    // cColProds = [ tar^3 - ~f1 * ~f2 * ~f3    = [ F1
    //               tar^3 -  f1 * ~f2 * ~f3        F2
    //               tar^3 -  f1 *  f2 * ~f3        F3
    //               tar^3 -  f1 *  f2 *  f3 ]      F4 ]
    Eigen::Map<Eigen::VectorXd> cColProds =
        createEmptyEigenVectorMap(_m_cColProds, n + 1);
    double ret = helperMonotonicDNF(cColProds, fxMat);
    // ecColProds = [ F2 * F3 * F4
    //                F1 * F3 * F4
    //                F1 * F2 * F4
    //                F1 * F2 * F3 ]
    Eigen::Map<Eigen::VectorXd> ecColProds =
        createEmptyEigenVectorMap(_m_ecColProds, n + 1);
    ecColProds(0) = cColProds.segment(1, n).prod();
    ecColProds(n) = cColProds.segment(0, n).prod();
    for (int i = 1; i < n; ++i) {
      ecColProds(i) = cColProds.head(i).prod() *
          cColProds.tail(n - i).prod();
    }
    // gpMat: d-by-n =
    //   f1' * [ -~f2 * ~f3 * F2 * F3 * F4 + ~f2 * ~f3 * F1 * F3 * F4
    //           + f2 * ~f3 * F1 * F2 * F4 +  f2 *  f3 * F1 * F2 * F3 ]
    // + f2' * [ -~f1 * ~f3 * F2 * F3 * F4 -  f1 * ~f3 * F1 * F3 * F4
    //           + f1 * ~f3 * F1 * F2 * F4 +  f1 *  f3 * F1 * F2 * F3 ]
    // + f3' * [ -~f1 * ~f2 * F2 * F3 * F4 -  f1 * ~f2 * F1 * F3 * F4
    //           - f1 *  f2 * F1 * F2 * F4 +  f1 *  f2 * F1 * F2 * F3 ]
    Eigen::Map<Eigen::MatrixXd> gpMat =
        createEmptyEigenMatrixMap(_m_gpMat, d, n);
    Eigen::Map<Eigen::MatrixXd> efxMat =
        createEmptyEigenMatrixMap(_m_efxMat, n - 1, n + 1);
    for (int i = 0; i < n; ++i) {
      // An example: i = 1
      //   efxMat = [ ~f1  f1  f1  f1
      //              ~f3 ~f3 ~f3  f3 ]
      efxMat.topRows(i) = fxMat.topRows(i);
      efxMat.bottomRows(n - 1 - i) = fxMat.bottomRows(n - 1 - i);
      //   efxColProds = [ -~f1 * ~f3
      //                   - f1 * ~f3
      //                     f1 * ~f3
      //                     f1 *  f3 ]
      Eigen::Map<Eigen::VectorXd> efxColProds =
          createEmptyEigenVectorMap(_m_efxColProds, n + 1);
      efxColProds = efxMat.colwise().prod();
      for (int j = 0; j <= i; ++j) { efxColProds(j) = -efxColProds(j); }
      gpMat.col(i) = gxMat.col(i) * ecColProds.dot(efxColProds);
    }
    Eigen::Map<Eigen::VectorXd>(g, d) = gpMat.rowwise().sum();
    return ret;
  }

  friend Eigen::Map<Eigen::VectorXd>
  createEmptyEigenVectorMap (std::vector<double>& mem, int d);

  friend Eigen::Map<Eigen::MatrixXd>
  createEmptyEigenMatrixMap (std::vector<double>& mem, int r, int c);

  std::vector<double> _m_fxMat;
  std::vector<double> _m_gxMat;
  std::vector<double> _m_cColProds;
  std::vector<double> _m_ecColProds;
  std::vector<double> _m_gpMat;
  std::vector<double> _m_efxMat;
  std::vector<double> _m_efxColProds;
};


template <typename CFunc>
class UniqueDNF : public opt::TFunction<Eigen::MatrixXd> {
 public:
  typedef opt::TFunction<Eigen::MatrixXd> Super;
  typedef UniqueDNF<CFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef CFunc Classifier;

  // Classifier function must be a batch function
  std::shared_ptr<Classifier> f;
  double target;

  virtual void initialize (
      std::shared_ptr<Classifier> f_, double target_) {
    f = f_;
    target = target_;
  }

  UniqueDNF () {}

  UniqueDNF (std::shared_ptr<Classifier> f, double target)
  { initialize(f, target); }

  ~UniqueDNF () override {}

  virtual void scopy (Self const& x) {
    if (!f || f == x.f) { f = std::make_shared<Classifier>(); }
    f->scopy(*x.f);
    target = x.target;
  }

  double operator() (Eigen::MatrixXd const& x) override
  { return helperUniqueDNF(x, *f); }

  double operator() (double* g, Eigen::MatrixXd const& x) override
  { return helperUniqueDNF(g, x, *f); }

  int dim () const override { return f->dim(); }

  void update (double const* w, int d) override { f->update(w, d); }

  double* data () override { return f->data(); }

  double const* data () const override { return f->data(); }

 protected:
  // fxMat: n-by-n matri, diagonal should be initialized
  //   Example: n = 3
  //     Input:    fxMat = [  f1   -   -
  //                           -  f2   -
  //                           -   -  f3  ]
  //     Output:   fxMat = [  f1 ~f1 ~f1
  //                         ~f2  f2 ~f2
  //                         ~f3 ~f3  f3  ]
  //               cColProds = [ tar^3 -  f1 * ~f2 * ~f3
  //                             tar^3 - ~f1 *  f2 * ~f3
  //                             tar^3 - ~f1 * ~f2 *  f3 ]
  // return: 1 - cColProds.prod()
  double helperUniqueDNF (
      Eigen::Map<Eigen::VectorXd>& cColProds,
      Eigen::Map<Eigen::MatrixXd>& fxMat) {
    int n = fxMat.rows();
    for (int i = 0; i < n; ++i) {
      double nfx = 1.0 - fxMat(i, i);
      fxMat.row(i).head(i).fill(nfx);
      fxMat.row(i).tail(n - i - 1).fill(nfx);
    }
    cColProds = std::pow(target, n) - fxMat.colwise().prod().array();
    return 1.0 - cColProds.prod();
  }

  // x: d-by-n sample matrix
  // f: function applied to x; should return n-by-1 vector
  template <typename Func> double
  helperUniqueDNF (Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    if (n == 0) { return 0.0; }
    Eigen::Map<Eigen::MatrixXd> fxMat =
        createEmptyEigenMatrixMap(_m_fxMat, n, n);
    fxMat.diagonal() = f(x);
    Eigen::Map<Eigen::VectorXd> cColProds =
        createEmptyEigenVectorMap(_m_cColProds, n);
    return helperUniqueDNF(cColProds, fxMat);
  }

  template <typename Func> double
  helperUniqueDNF (double* g, Eigen::MatrixXd const& x, Func& f) {
    int n = x.cols();
    int d = f.dim();
    // fxMat = [  f1 ~f1 ~f1
    //           ~f2  f2 ~f2
    //           ~f3 ~f3  f3  ]
    Eigen::Map<Eigen::MatrixXd> fxMat =
        createEmptyEigenMatrixMap(_m_fxMat, n, n);
    // gxMat = [ f1' f2' f3' ]
    Eigen::Map<Eigen::MatrixXd> gxMat =
        createEmptyEigenMatrixMap(_m_gxMat, d, n);
    fxMat.diagonal() = f(gxMat.data(), x);
    // Complement column products: n-by-1
    //   cColProds = [ tar^3 -  f1 * ~f2 * ~f3   = [ F1
    //                 tar^3 - ~f1 *  f2 * ~f3   =   F2
    //                 tar^3 - ~f1 * ~f2 *  f3 ] =   F3 ]
    Eigen::Map<Eigen::VectorXd> cColProds =
        createEmptyEigenVectorMap(_m_cColProds, n);
    double ret = helperUniqueDNF(cColProds, fxMat);
    // ecColProds = [ F2 * F3
    //                F1 * F3
    //                F1 * F2 ]
    Eigen::Map<Eigen::VectorXd> ecColProds =
        createEmptyEigenVectorMap(_m_ecColProds, n);
    ecColProds(0) = cColProds.segment(1, n - 1).prod();
    ecColProds(n - 1) = cColProds.segment(0, n - 1).prod();
    for (int i = 1; i < n - 1; ++i) {
      ecColProds(i) = cColProds.head(i).prod() *
          cColProds.tail(n - i - 1).prod();
    }
    // gpMat: d-by-n =
    //   f1' * [  ~f2 * ~f3 * F2 * F3 +
    //           - f2 * ~f3 * F1 * F3 +
    //           -~f2 *  f3 * F1 * F2 ]
    // + f2' * [ - f1 * ~f3 * F2 * F3 +
    //            ~f1 * ~f3 * F1 * F3 +
    //           -~f1 *  f3 * F1 * F2 ]
    // + f3' * [ - f1 * ~f2 * F2 * F3 +
    //           -~f1 *  f2 * F1 * F3 +
    //           -~f1 * ~f2 * F1 * F2 ]
    Eigen::Map<Eigen::MatrixXd> gpMat =
        createEmptyEigenMatrixMap(_m_gpMat, d, n);
    Eigen::Map<Eigen::MatrixXd> efxMat =
        createEmptyEigenMatrixMap(_m_efxMat, n - 1, n);
    for (int i = 0; i < n; ++i) {
      // An example: i = 1
      //   efxMat = [  f1 ~f1 ~f1
      //              ~f3 ~f3  f3 ]
      efxMat.topRows(i) = fxMat.topRows(i);
      efxMat.bottomRows(n - 1 - i) = fxMat.bottomRows(n - 1 - i);
      //   efxColProds = [ - f1 * ~f3
      //                    ~f1 * ~f3
      //                   -~f1 *  f3 ]
      Eigen::Map<Eigen::VectorXd> efxColProds =
          createEmptyEigenVectorMap(_m_efxColProds, n);
      efxColProds = -efxMat.colwise().prod();
      efxColProds(i) = -efxColProds(i);
      gpMat.col(i) = gxMat.col(i) * ecColProds.dot(efxColProds);
    }
    Eigen::Map<Eigen::VectorXd>(g, d) = gpMat.rowwise().sum();
    return ret;
  }

  friend Eigen::Map<Eigen::VectorXd>
  createEmptyEigenVectorMap (std::vector<double>& mem, int d);

  friend Eigen::Map<Eigen::MatrixXd>
  createEmptyEigenMatrixMap (std::vector<double>& mem, int r, int c);

  std::vector<double> _m_fxMat;
  std::vector<double> _m_gxMat;
  std::vector<double> _m_cColProds;
  std::vector<double> _m_ecColProds;
  std::vector<double> _m_gpMat;
  std::vector<double> _m_efxMat;
  std::vector<double> _m_efxColProds;
};

};
};

#endif
