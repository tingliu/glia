#include "util/text_cmd.hxx"
#include "sshmt_util.hxx"
#include "util/text_io.hxx"
using namespace glia;
using namespace glia::sshmt;

#define EIGEN_DONT_PARALLELIZE

bool verbose = true;
std::string inputModelFile;
std::vector<std::string> supFeatFiles;
std::vector<std::string> supLabelFiles;
std::vector<std::string> unsFeatFiles;
std::vector<std::string> unsOrderFiles;
int unsBatchSize = -1;
int supBatchSize = -1;
bool balSupBatch = false;
double wr = 0.0;
double ws = 0.0;
double wu = 0.0;
double sigmaS = 0.0;
bool doUpdateSigmaS = false;
double sigmaU = 0.0;
bool doUpdateSigmaU = false;
int optimizerType = 1;  // 1: Vanilla, 2: Adam, 3: Momentum
double step = 0.0;
double stepDecay = 1.0;
int fixedStepDecayIterations = -1;
int nEpochPerIteration = 1;
int nBatchPerIteration = 1;
int maxIterations = 0;
int nSigmaUpdate = 0;
int fixedStepDecaySigmaUpdatePeriod = -1;
int saveInterval = INT_MAX;
std::string outputModelFilePattern;

double minSigma2 = 1e-6;
double pathTarget = 1.0;
double mergeTarget = 0.95;
int maxPathLength = 3;
int minPathLength = 2;
double posLabelTarget = 0.05;
double negLabelTarget = 0.95;
std::unordered_map<int, double> labelTargetMap{
  {1, posLabelTarget}, {-1, negLabelTarget}};

typedef SshmtInput Input;
typedef alg::Logsig Classifier;
typedef MonotonicDnfGaussian<Classifier, Input> EnergyFunction;
typedef opt::TOptimizer<EnergyFunction> Optimizer;

void printParams () {
  disp("minSigma2 = %g", minSigma2);
  disp("pathTarget = %g", pathTarget);
  disp("mergeTarget = %g", mergeTarget);
  disp("maxPathLength = %d", maxPathLength);
  disp("minPathLength = %d", minPathLength);
  disp("posLabelTarget = %g", posLabelTarget);
  disp("negLabelTarget = %g", negLabelTarget);
  std::string str;
  for (auto const& lt: labelTargetMap)
  { str += strprintf("(%d: %g) ", lt.first, lt.second); }
  disp("labelTargetMap = {%s}", str.c_str());
  disp("----------------");
  disp("unsBatchSize = %d", unsBatchSize);
  disp("supBatchSize = %d", supBatchSize);
  disp("balSupBatch = %d", balSupBatch);
  disp("wr = %g", wr);
  disp("ws = %g", ws);
  disp("wu = %g", wu);
  disp("sigmaS = %g", sigmaS);
  disp("doUpdateSigmaS = %d", doUpdateSigmaS);
  disp("sigmaU = %g", sigmaU);
  disp("doUpdateSigmaU = %d", doUpdateSigmaU);
  disp("optimizerType = %d", optimizerType);
  disp("step = %g", step);
  disp("stepDecay = %g", stepDecay);
  disp("fixedStepDecayIterations = %d", fixedStepDecayIterations);
  disp("nEpochPerIteration = %d", nEpochPerIteration);
  disp("nBatchPerIteration = %d", nBatchPerIteration);
  disp("maxIterations = %d", maxIterations);
  disp("nSigmaUpdate = %d", nSigmaUpdate);
  disp("fixedStepDecaySigmaUpdatePeriod = %d",
       fixedStepDecaySigmaUpdatePeriod);
  disp("saveInterval = %d", saveInterval);
  disp("----------------");
}


std::shared_ptr<EnergyFunction> prepareEnergyFunction (
    std::shared_ptr<Input> input, std::string const& inputModelFile)
{
  auto classifier = std::make_shared<Classifier>(input->dim());
  if (inputModelFile.empty()) {
    classifier->w->setZero();
  } else {
    if (verbose) { disp("Initializing from input model..."); }
    std::vector<double> w;
    readData(w, inputModelFile, true);
    classifier->update(w.data(), w.size());
  }
  return std::make_shared<EnergyFunction>(
      classifier, wr, wu, ws, mergeTarget, sigmaU, sigmaS);
}


bool operation ()
{
  printParams();
  auto input = prepareInput(
      supLabelFiles, supFeatFiles, unsOrderFiles, unsFeatFiles, true,
      supBatchSize, balSupBatch, unsBatchSize, posLabelTarget,
      negLabelTarget, maxPathLength, minPathLength, pathTarget,
      labelTargetMap, verbose);
  auto energyFunction = prepareEnergyFunction(input, inputModelFile);
  auto optimizer = prepareOptimizer(
      energyFunction, input, optimizerType, step, stepDecay,
      fixedStepDecayIterations, maxIterations, nEpochPerIteration,
      nBatchPerIteration, verbose);
  // // Trial: alternate optimization
  // optimizer->param.altSU = true;
  // // ~ Trial: alternate optimization
  double& sigmaS2 = optimizer->pFunc->supervised->sigma2;
  double& sigmaU2 = optimizer->pFunc->unsupervised->sigma2;
  double sigmaS2Eval = 0.0, sigmaU2Eval = 0.0;
  updateSigmas(
      optimizer, doUpdateSigmaS, doUpdateSigmaU, sigmaS2, sigmaS2Eval,
      sigmaU2, sigmaU2Eval, 0.0, verbose);
  if (verbose) {
    disp("\tlearn-%d: su = %g (%g), ss = %g (%g)", 0, std::sqrt(sigmaU2),
         std::sqrt(sigmaU2Eval), std::sqrt(sigmaS2),
         std::sqrt(sigmaS2Eval));
  }
  for (int i = 0; i < nSigmaUpdate; ++i) {
    if (fixedStepDecaySigmaUpdatePeriod > 0 && i > 0 &&
        i % fixedStepDecaySigmaUpdatePeriod == 0) {
      // Cut learning rate to half
      optimizer->param.step = std::max(
          optimizer->param.step * 0.5, optimizer->param.minStep);
    }
    optimizer->run();
    if (!outputModelFilePattern.empty() &&
        (((i + 1) % saveInterval == 0) || i == nSigmaUpdate - 1)) {
      writeData(
          strprintf(outputModelFilePattern.c_str(), i),
          convertFromEigenMatrix(*optimizer->pFunc->w), "\n");
    }
    updateSigmas(
        optimizer, doUpdateSigmaS, doUpdateSigmaU, sigmaS2, sigmaS2Eval,
        sigmaU2, sigmaU2Eval, 0.0, verbose);
    if (verbose) {
      disp("\tlearn-%d: su = %g (%g), ss = %g (%g)", i + 1,
           std::sqrt(sigmaU2), std::sqrt(sigmaU2Eval),
           std::sqrt(sigmaS2), std::sqrt(sigmaS2Eval));
    }
  }
  return true;
}


int main (int argc, char* argv[])
{
  Eigen::initParallel();
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("im", bpo::value<std::string>(&inputModelFile), "Input model file")
      ("v", bpo::value<bool>(&verbose), "Whether to print info "
       "[default: true]")
      ("sl", bpo::value<std::vector<std::string>>(&supLabelFiles),
       "Input supervised label file(s)")
      ("sf", bpo::value<std::vector<std::string>>(&supFeatFiles),
       "Input supervised feature file(s)")
      ("uo", bpo::value<std::vector<std::string>>(&unsOrderFiles),
       "Input unsupervised merge order file(s)")
      ("uf", bpo::value<std::vector<std::string>>(&unsFeatFiles),
       "Input unsupervised feature file(s)")
      ("bsu", bpo::value<int>(&unsBatchSize),
       "Unsupervised data batch size (Use -1 to ignore) [default: -1]")
      ("bss", bpo::value<int>(&supBatchSize),
       "Supervised data batch size (Use -1 to ignore) [default: -1]")
      ("bb", bpo::value<bool>(&balSupBatch),
       "Whether to balance supervised data batch [default: false]")
      ("wr", bpo::value<double>(&wr), "Weight for regularizer term "
       "(Use 0.0 to bypass) [default: 0.0]")
      ("ws", bpo::value<double>(&ws), "Weight for supervised term "
       "(Use 0.0 to bypass) [default: 0.0]")
      ("wu", bpo::value<double>(&wu), "Weight for unsupervised term "
       "(Use 0.0 to bypass) [default: 0.0]")
      ("ss", bpo::value<double>(&sigmaS), "Sigma for supervised term "
       "(Used only if fixed) [default: 0.0]")
      ("su", bpo::value<double>(&sigmaU), "Sigma for unsupervised term "
       "(Used only if fixed) [default: 0.0]")
      ("uss", bpo::value<bool>(&doUpdateSigmaS), "Whether to update "
       "sigma for supervised term [default: false]")
      ("usu", bpo::value<bool>(&doUpdateSigmaU), "Whether to update "
       "sigma for unsupervised term [default: false]")
      ("topt", bpo::value<int>(&optimizerType), "Optimizer type: "
       "1: Vanilla, 2: Adam, 3: Momentum [default: 1]")
      ("lr", bpo::value<double>(&step), "Gradient descent step size "
       "[default: 0.0]")
      ("lrd", bpo::value<double>(&stepDecay), "Gradient descent step "
       "size decay (Use 1.0 for fixed step GD) [default: 1.0]")
      ("lrdi", bpo::value<int>(&fixedStepDecayIterations),
       "Period (iterations) of fixed step decay "
       "(Only used if stepDecay != 1.0, use -1 to bypass) [default: -1]")
      ("lrds", bpo::value<int>(&fixedStepDecaySigmaUpdatePeriod),
       "Period (sigma updates) of fixed step decay (Use -1 to bypass) "
       "[default: -1]")
      ("te", bpo::value<int>(&nEpochPerIteration), "Number of epochs "
       "in each iteration [default: 1]")
      ("tb", bpo::value<int>(&nBatchPerIteration), "Number of batches "
       "in each iteration (Only used if nEpochPerIteration <= 0) "
       "[default: 1]")
      ("tw", bpo::value<int>(&maxIterations), "Maximum iterations "
       "[default: 0]")
      ("ts", bpo::value<int>(&nSigmaUpdate), "Number of sigma updates "
       "[default: 0]")
      ("si", bpo::value<int>(&saveInterval), "Iteration interval to "
       "save model(s) [default: INT_MAX]")
      ("omp", bpo::value<std::string>(&outputModelFilePattern)->required(),
       "Output model file pattern");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
