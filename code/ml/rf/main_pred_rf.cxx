#include "alg/rf.hxx"
#include "alg/function.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> modelFiles;
std::vector<double> modelDistributorArgs;
std::vector<std::string> featFiles;
int predictLabel;
std::vector<std::string> predFiles;

bool operation ()
{
  // Prepare classifier
  std::shared_ptr<opt::TFunction<std::vector<FVal>>> rf;
  if (modelFiles.size() == 1) {
    rf = std::make_shared<alg::RandomForest>(
        predictLabel, modelFiles.front());
  } else {
    if (modelDistributorArgs.size() != 3)
    { perr("Error: model distributor needs 3 arguments..."); }
    rf = std::make_shared<alg::EnsembleRandomForest>(
        predictLabel, modelFiles, opt::ThresholdModelDistributor<FVal>(
            modelDistributorArgs[0], modelDistributorArgs[1],
            modelDistributorArgs[2]));
  }
  // Predict
  std::vector<std::vector<FVal>> feats;
  std::vector<double> preds;
  for (int i = 0; i < featFiles.size(); ++i) {
    feats.clear();
    preds.clear();
    readData(feats, featFiles[i]);
    preds.reserve(feats.size());
    for (auto const& x : feats) { preds.push_back(rf->operator()(x)); }
    writeData(predFiles[i], preds, "\n", FLT_PREC);
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("m", bpo::value<std::vector<std::string>>(&modelFiles)->required(),
       "Input model file name(s)")
      ("md",
       bpo::value<std::vector<double>>(&modelDistributorArgs)->multitoken(),
       "Model distributor argument(s) (e.g. 0 1 areaMedian)")
      ("f", bpo::value<std::vector<std::string>>(&featFiles)->required(),
       "Input feature file name(s)")
      ("l", bpo::value<int>(&predictLabel)->required(),
       "Label to predict (only supporting binary prediction)")
      ("p", bpo::value<std::vector<std::string>>(&predFiles)->required(),
       "Output prediction file name(s)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
