#ifndef _glia_sshmt_input_hxx_
#define _glia_sshmt_input_hxx_

#include "type/energy_function.hxx"
#include "util/container.hxx"
#include "util/mp.hxx"

namespace glia {
namespace sshmt {

class PathInput
    : public opt::FunctionInputSampleTarget<Eigen::MatrixXd, double> {
 public:
  typedef opt::FunctionInputSampleTarget<Eigen::MatrixXd, double> Super;
  typedef PathInput Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef Super::Sample Sample;
  typedef Super::Target Target;

  void initialize (
      std::vector<std::vector<std::vector<int>>> const& paths,
      std::vector<std::vector<std::vector<FVal>>> const& samples_,
      double target) {
    if (paths.empty()) { return; }
    std::vector<std::pair<int, int>> indexMap;
    for (int i = 0; i < paths.size(); ++i) {
      for (int j = 0; j < paths[i].size(); ++j)
      { indexMap.emplace_back(std::make_pair(i, j)); }
    }
    int d = samples_.front().front().size();
    int n = indexMap.size();
    Super::_targets->reserve(n);
    Super::_samples->reserve(n);
    for (int i = 0; i < n; ++i) {
      int pathLength =
          paths[indexMap[i].first][indexMap[i].second].size();
      Super::_targets->emplace_back(std::make_shared<Target>(
          std::pow(target, pathLength)));
      Super::_samples->emplace_back(std::make_shared<Sample>(
          d, pathLength));
    }
    parfor(0, n, true, [this, &indexMap, &paths, &samples_](int i) {
        int j = indexMap[i].first;
        int k = indexMap[i].second;
        convertToEigenMatrix(*_samples->at(i), samples_[j], paths[j][k]);
      }, 0);
  }

  PathInput () {}

  PathInput (
      std::vector<std::vector<std::vector<int>>> const& paths,
      std::vector<std::vector<std::vector<FVal>>> const& samples_,
      double target) { initialize(paths, samples_, target); }

  ~PathInput () override {}

  int dim () const override { return Super::_samples->front()->rows(); }
};


class SampleInput
    : public opt::FunctionInputSampleTarget<Eigen::VectorXd, double> {
 public:
  typedef
  opt::FunctionInputSampleTarget<Eigen::VectorXd, double> Super;
  typedef SampleInput Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef Super::Sample Sample;
  typedef Super::Target Target;

  void initialize (
      std::vector<std::vector<std::vector<FVal>>> const& samples_,
      std::vector<std::vector<int>> const& labels,
      std::unordered_map<int, double> const& labelToTargetMap) {
    if (labels.empty()) { return; }
    int d = samples_.front().front().size();
    for (int j = 0; j < labels.size(); ++j) {
      for (int k = 0; k < labels[j].size(); ++k) {
        auto ltit = labelToTargetMap.find(labels[j][k]);
        if (ltit != labelToTargetMap.end()) {  // Valid label
          Super::_targets->emplace_back(
              std::make_shared<Target>(ltit->second));
          Super::_samples->emplace_back(std::make_shared<Sample>(d));
          convertToEigenMatrix(*Super::_samples->back(), samples_[j][k]);
        }
      }
    }
  }

  SampleInput () {}

  SampleInput (
      std::vector<std::vector<std::vector<FVal>>> const& samples_,
      std::vector<std::vector<int>> const& labels,
      std::unordered_map<int, double> const& labelToTargetMap)
  { initialize(samples_, labels, labelToTargetMap); }

  ~SampleInput () override {}

  int dim () const override { return Super::_samples->front()->size(); }
};


class SshmtInput
    : public opt::TEnergyFunctionInput<PathInput, SampleInput> {
 public:
  typedef TEnergyFunctionInput<PathInput, SampleInput> Super;
  typedef SshmtInput Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  virtual void initialize (
      std::shared_ptr<PathInput> pPathInput,
      std::shared_ptr<SampleInput> pSampleInput) {
    Super::unsupervised = pPathInput;
    Super::supervised = pSampleInput;
    if (Super::unsupervised->allSize() == 0)
    { Super::useUnsupervised = false; }
    if (Super::supervised->allSize() == 0)
    { Super::useSupervised = false; }
  }

  SshmtInput () {}

  SshmtInput (
      std::shared_ptr<PathInput> pPathInput,
      std::shared_ptr<SampleInput> pSampleInput)
  { initialize(pPathInput, pSampleInput); }

  ~SshmtInput () override {}
};

};

};

#endif
