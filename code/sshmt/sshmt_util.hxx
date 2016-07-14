#include "sshmt/energy_function.hxx"
#include "sshmt/input.hxx"
#include "hmt/tree_build.hxx"
#include "alg/function.hxx"
#include "alg/gd.hxx"
#include "util/text_io.hxx"

namespace glia {
namespace sshmt {

std::shared_ptr<SampleInput> prepareSampleInput (
    std::vector<std::string> const& labelFiles,
    std::vector<std::string> const& featFiles, bool doAppendBias,
    std::unordered_map<int, double> const& labelTargetMap) {
  int n = featFiles.size();
  std::vector<std::vector<std::vector<FVal>>> samples(n);
  std::vector<std::vector<int>> labels(n);
  parfor(
      0, n, true, [&samples, &labels, doAppendBias, &labelFiles,
                   &featFiles](int i) {
        readData(labels[i], labelFiles[i], true);
        readData(samples[i], featFiles[i]);
        if (doAppendBias)
        { appendData(samples[i], std::vector<FVal>(1, 1.0)); }
      }, 0);
  return std::make_shared<SampleInput>(samples, labels, labelTargetMap);
}


std::shared_ptr<PathInput> preparePathInput (
    std::vector<std::string> const& orderFiles,
    std::vector<std::string> const& featFiles,  bool doAppendBias,
    int maxPathLength, int minPathLength, double pathTarget) {
  int n = featFiles.size();
  std::vector<std::vector<std::vector<FVal>>> samples(n);
  std::vector<std::vector<std::vector<int>>> paths(n);
  parfor(
      0, n, true, [&samples, &paths, doAppendBias, &orderFiles,
                   &featFiles, maxPathLength, minPathLength,
                   pathTarget](int i) {
        std::vector<TTriple<Label>> order;
        readData(order, orderFiles[i], true);
        if (maxPathLength >= 0) {
          hmt::genMergePaths(paths[i], order, maxPathLength, minPathLength);
        } else { hmt::genMergePaths(paths[i], order); }
        readData(samples[i], featFiles[i]);
        if (doAppendBias)
        { appendData(samples[i], std::vector<FVal>(1, 1.0)); }
      }, 0);
  return std::make_shared<PathInput>(paths, samples, pathTarget);
}


std::shared_ptr<SshmtInput> prepareInput (
    std::vector<std::string> const& supLabelFiles,
    std::vector<std::string> const& supFeatFiles,
    std::vector<std::string> const& unsOrderFiles,
    std::vector<std::string> const& unsFeatFiles,
    bool doAppendBias, int supBatchSize, bool balSupBatch,
    int unsBatchSize, double posLabelTarget, double negLabelTarget,
    int maxPathLength, int minPathLength, double pathTarget,
    std::unordered_map<int, double> const& labelTargetMap,
    bool verbose) {
  if (verbose) { disp("Loading supervised data..."); }
  auto supInput = prepareSampleInput(
      supLabelFiles, supFeatFiles, doAppendBias, labelTargetMap);
  if (verbose)
  { disp("Use %d supervised samples.", supInput->allSize()); }
  if (supBatchSize > 0) {
    if (balSupBatch) {
      std::vector<std::pair<double, int>> targetBatchSize_;
      targetBatchSize_.push_back(std::make_pair(
          posLabelTarget, supBatchSize / 2));
      targetBatchSize_.push_back(std::make_pair(
          negLabelTarget, supBatchSize - supBatchSize / 2));
      supInput->setBatchSampler<ClassBatchSampler>(
          targetBatchSize_, supInput->allTargets());
    } else {
      supInput->setBatchSampler<UniformBatchSampler>(
          supInput->allSize(), supBatchSize);
    }
    if (verbose) {
      disp("Use %d-batch for supervised samples.",
           supInput->getBatchSampler()->getBatchSize());
    }
  } else if (balSupBatch) {  // Special: balance by using all minority
    supInput->setBatchSampler<ClassBatchSampler>(
        supInput->allTargets(), 1.0);
    if (verbose) {
      disp("Use %d-batch for supervised samples.",
           supInput->getBatchSampler()->getBatchSize());
    }
  }
  if (verbose) { disp("Loading unsupervised data..."); }
  auto unsInput = preparePathInput(
      unsOrderFiles, unsFeatFiles, doAppendBias, maxPathLength,
      minPathLength, pathTarget);
  if (verbose)
  { disp("Use %d unsupervised samples.", unsInput->allSize()); }
  if (unsBatchSize > 0) {
    unsInput->setBatchSampler<UniformBatchSampler>(
        unsInput->allSize(), unsBatchSize);
    if (verbose) {
      disp("Use %d-batch for unsupervised samples.",
           unsInput->getBatchSampler()->getBatchSize());
    }
  }
  return std::make_shared<SshmtInput>(unsInput, supInput);
}


template <typename EFunc> std::shared_ptr<opt::TGdOptimizer<EFunc>>
prepareOptimizer (
    std::shared_ptr<EFunc> energyFunction,
    std::shared_ptr<SshmtInput> input, int optimizerType, double step,
    double stepDecay, int fixedStepDecayIterations, int maxIterations,
    int nEpochPerIteration, int nBatchPerIteration, bool verbose)
{
  std::shared_ptr<opt::TGdOptimizer<EFunc>> ret;
  if (optimizerType == 1) {
    ret = std::make_shared<opt::TGdOptimizer<EFunc>>(
        energyFunction, input, verbose, step, stepDecay,
        fixedStepDecayIterations, maxIterations, nEpochPerIteration,
        nBatchPerIteration);
  } else if (optimizerType == 2) {
    ret = std::make_shared<opt::TAdamOptimizer<EFunc>>(
        energyFunction, input, verbose, step, stepDecay,
        fixedStepDecayIterations, maxIterations, nEpochPerIteration,
        nBatchPerIteration);
  } else if (optimizerType == 3) {
    ret = std::make_shared<opt::TMomentumOptimizer<EFunc>>(
        energyFunction, input, verbose, step, stepDecay,
        fixedStepDecayIterations, maxIterations, nEpochPerIteration,
        nBatchPerIteration);
  } else { perr("Error: unsupported optimizer type..."); }
  return ret;
}


template <typename Func, typename TInput> double
getSigma2 (Func& f, TInput const& x, double minSigma2)
{
  double ret = f.loss(x.allSamples(), x.allTargets()) / x.allSize();
  return std::max(minSigma2, ret);
}


template <typename TOptim>
void updateSigmaU (
    std::shared_ptr<TOptim> optimizer, bool doUpdateSigmaU,
    double& sigmaU2, double& sigmaU2Eval, double minSigma2, bool verbose)
{
  if (optimizer->pInput->useUnsupervised) {
    if (doUpdateSigmaU || !verbose) {
      // // Trial
      // DO_WEIRD_STUFF = 1;
      // // ~ Trial
      sigmaU2 = getSigma2(
          *optimizer->pFunc->unsupervised,
          *optimizer->pInput->unsupervised, minSigma2);
      // // Trial
      // DO_WEIRD_STUFF = 0;
      // // ~ Trial
      // // Trial
      // DO_WEIRD_STUFF = 1;
      // // ~ Trial
      sigmaU2Eval = sigmaU2 == minSigma2 ? sigmaU2 : getSigma2(
          *optimizer->pFunc->unsupervised,
          *optimizer->pInput->unsupervised, 0.0);
      // // Trial
      // DO_WEIRD_STUFF = 0;
      // // ~ Trial
    } else {
      // // Trial
      // DO_WEIRD_STUFF = 1;
      // // ~ Trial
      sigmaU2Eval = getSigma2(
          *optimizer->pFunc->unsupervised,
          *optimizer->pInput->unsupervised, 0.0);
      // // Trial
      // DO_WEIRD_STUFF = 0;
      // // ~ Trial
    }
  }
}


template <typename TOptim>
void updateSigmaS (
    std::shared_ptr<TOptim> optimizer, bool doUpdateSigmaS,
    double& sigmaS2, double& sigmaS2Eval, double minSigma2, bool verbose)
{
  if (optimizer->pInput->useSupervised) {
    if (doUpdateSigmaS || !verbose) {
      sigmaS2 = getSigma2(
          *optimizer->pFunc->supervised, *optimizer->pInput->supervised,
          minSigma2);
      sigmaS2Eval = sigmaS2 != minSigma2 ? sigmaS2 : getSigma2(
          *optimizer->pFunc->supervised, *optimizer->pInput->supervised,
          0.0);
    } else {
      sigmaS2Eval = getSigma2(
          *optimizer->pFunc->supervised, *optimizer->pInput->supervised,
          0.0);
    }
  }
}


template <typename TOptim>
void updateSigmas (
    std::shared_ptr<TOptim> optimizer, bool doUpdateSigmaS,
    bool doUpdateSigmaU, double& sigmaS2, double& sigmaS2Eval,
    double& sigmaU2, double& sigmaU2Eval, double minSigma2, bool verbose)
{
  updateSigmaS(
      optimizer, doUpdateSigmaS, sigmaS2, sigmaS2Eval, minSigma2,
      verbose);
  updateSigmaU(
      optimizer, doUpdateSigmaU, sigmaU2, sigmaU2Eval, minSigma2,
      verbose);
}



};
};
