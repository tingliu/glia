#include "util/text_cmd.hxx"
#include "alg/nn.hxx"
#include "util/stats.hxx"
#include "util/mp.hxx"
#include "util/text_io.hxx"
using namespace glia;

#define EIGEN_DONT_PARALLELIZE

std::vector<std::string> modelFiles;
std::string featMinMaxFile;
std::vector<double> bcModelDistributorArgs;
std::vector<std::string> featFiles;
int nNodeLayer1;
int nNodeLayer2;
std::vector<std::string> predFiles;

bool operation ()
{
  std::vector<std::vector<FVal>> featMinMax;
  if (!featMinMaxFile.empty()) { readData(featMinMax, featMinMaxFile); }
  std::shared_ptr<opt::TFunction<std::vector<FVal>>> f;
  if (modelFiles.size() == 1) {
    f = std::make_shared<alg::MLP2v>(
        nNodeLayer1, nNodeLayer2, modelFiles.front());
  } else {
    f = std::make_shared<alg::EnsembleMLP2v>(
        nNodeLayer1, nNodeLayer2, modelFiles,
        opt::ThresholdModelDistributor<FVal>(
            bcModelDistributorArgs[0], bcModelDistributorArgs[1],
            bcModelDistributorArgs[2]));
  }
  std::vector<double> preds;
  for (int i = 0; i < featFiles.size(); ++i) {
    std::vector<std::vector<FVal>> samples;
    readData(samples, featFiles[i]);
    int n = samples.size();
    preds.resize(n);
    parfor(0, n, true, [&featMinMax, &preds, &samples, &f](int i) {
        if (!featMinMax.empty())
        { stats::rescale(samples[i], featMinMax, -1.0, 1.0); }
        samples[i].push_back(1.0);  // Do not forget to append 1 for bias
        preds[i] = f->operator()(samples[i]);
      }, 0);
    writeData(predFiles[i], preds, "\n");
  }
  return true;
}


int main (int argc, char* argv[])
{
  Eigen::initParallel();
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("m", bpo::value<std::vector<std::string>>(&modelFiles)->required(),
       "Trained model file(s)")
      ("fmm", bpo::value<std::string>(&featMinMaxFile),
       "Input feature min/max file name (ignore to not rescale)")
      ("md", bpo::value<std::vector<double>>(&bcModelDistributorArgs),
       "Boundary classifier distributor argument(s) "
       "(e.g. --md 0 --md 1 --md areaMedian)")
      ("f", bpo::value<std::vector<std::string>>(&featFiles)->required(),
       "Input feature files")
      ("nn1", bpo::value<int>(&nNodeLayer1)->required(),
       "Number of nodes in hidden layer 1")
      ("nn2", bpo::value<int>(&nNodeLayer2)->required(),
       "Number of nodes in hidden layer 2")
      ("p", bpo::value<std::vector<std::string>>(&predFiles)->required(),
       "Output prediction files");
  return parse(argc, argv, opts) && operation()? EXIT_SUCCESS: EXIT_FAILURE;
}
