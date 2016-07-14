#include "util/text_cmd.hxx"
#include "alg/function.hxx"
#include "util/text_io.hxx"
using namespace glia;

#define EIGEN_DONT_PARALLELIZE

std::string modelFile;
std::vector<std::string> featFiles;
std::vector<std::string> predFiles;

bool operation ()
{
  std::vector<double> w;
  readData(w, modelFile, true);
  alg::Logsig f(w.size());
  f.update(w.data(), w.size());
  std::vector<double> preds;
  for (int i = 0; i < featFiles.size(); ++i) {
    std::vector<std::vector<FVal>> samples;
    readData(samples, featFiles[i]);
    // Do not forget to append 1 for bias
    appendData(samples, std::vector<FVal>(1, 1.0));
    int n = samples.size();
    preds.resize(n);
    parfor(0, n, true, [&preds, &samples, &f](int i) {
        preds[i] = f(Eigen::Map<Eigen::VectorXd>(
            samples[i].data(), samples[i].size())); }, 0);
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
      ("m", bpo::value<std::string>(&modelFile)->required(),
       "Trained model file")
      ("f", bpo::value<std::vector<std::string>>(&featFiles)->required(),
       "Input feature files")
      ("p", bpo::value<std::vector<std::string>>(&predFiles)->required(),
       "Output prediction files");
  return parse(argc, argv, opts) && operation()? EXIT_SUCCESS: EXIT_FAILURE;
}
