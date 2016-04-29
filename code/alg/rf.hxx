#ifndef _glia_alg_rf_hxx_
#define _glia_alg_rf_hxx_

#include "type/function.hxx"
#include "ml/rf/rf.hxx"

namespace glia {
namespace alg {

class RandomForest : public virtual opt::TFunction<std::vector<FVal>> {
 public:
  typedef opt::TFunction<std::vector<FVal>> Super;
  typedef RandomForest Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef std::vector<FVal> Input;

  int predictLabel;
  std::shared_ptr<ml::rf::Model> model;

  virtual void initialize (
      int predictLabel_, std::string const& modelFile) {
    predictLabel = predictLabel_;
    model = std::make_shared<ml::rf::Model>();
    model->readFromOld(modelFile);
  }

  RandomForest () {}

  RandomForest (int predictLabel, std::string const& modelFile)
  { initialize(predictLabel, modelFile); }

  ~RandomForest () override {}

  double operator() (Input const& x) override
  { return model->predict((double*)x.data(), x.size(), predictLabel); }

  double operator() (double* g, Input const& x) override {
    perr("Error: no gradient available for random forest...");
    return operator()(x);
  }

  int dim () const override {
    perr("Error: no dimension available for random forest...");
    return -1;
  }

  void update (double const* w_, int d) override
  { perr("Error: no update can be done to random forest..."); }

  double* data () override {
    perr("Error: no data pointer can be returned for random forest...");
    return nullptr;
  }

  double const* data () const override {
    perr("Error: no data pointer can be returned for random forest...");
    return nullptr;
  }
};


class EnsembleRandomForest
    : public virtual opt::TFunction<std::vector<FVal>> {
 public:
  typedef opt::TFunction<std::vector<FVal>> Super;
  typedef EnsembleRandomForest Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef std::vector<FVal> Input;
  typedef std::function<int(Input)> Distributor;

  std::vector<std::shared_ptr<RandomForest>> models;
  Distributor fdist;

  virtual void initialize (
      int predictLabel, std::vector<std::string> const& modelFiles,
      Distributor const& fdist_) {
    int n = modelFiles.size();
    models.resize(n);
    for (int i = 0; i < n; ++i) {
      models[i] = std::make_shared<RandomForest>(
          predictLabel, modelFiles[i]);
    }
    fdist = fdist_;
  }

  EnsembleRandomForest () {}

  EnsembleRandomForest (
      int predictLabel, std::vector<std::string> const& modelFiles,
      Distributor const& fdist)
  { initialize(predictLabel, modelFiles, fdist); }

  ~EnsembleRandomForest () override {}

  double operator() (Input const& x) override
  { return models[fdist(x)]->operator()(x); }

  double operator() (double* g, Input const& x) override {
    perr("Error: no gradient available for random forest...");
    return operator()(x);
  }

  int dim () const override {
    perr("Error: no dimension available for random forest...");
    return -1;
  }

  void update (double const* w_, int d) override
  { perr("Error: no update can be done to random forest..."); }

  double* data () override {
    perr("Error: no data pointer can be returned for random forest...");
    return nullptr;
  }

  double const* data () const override {
    perr("Error: no data pointer can be returned for random forest...");
    return nullptr;
  }
};

};
};

#endif
