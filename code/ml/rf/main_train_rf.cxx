#include "ml/rf/ml_rf.h"
#include "glia_base.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> featFiles;
std::vector<std::string> labelFiles;
std::string featFileNameFile;
std::string labelFileNameFile;
int nTree = 255;
int mTry = 0;
double sampleSizeRatio = 0.7;
int nodeSize = 1;
bool balance = true;
std::string modelFile;

bool operation ()
{
  std::vector<const char*> ffs, lfs;
  for (auto const& fn : featFiles) { ffs.push_back(fn.c_str()); }
  for (auto const& fn : labelFiles) { lfs.push_back(fn.c_str()); }
  if (!featFileNameFile.empty()) {
    std::vector<std::string> _featFiles, _labelFiles;
    readData(_featFiles, featFileNameFile, true);
    readData(_labelFiles, labelFileNameFile, true);
    for (auto const& fn : _featFiles) { ffs.push_back(fn.c_str()); }
    for (auto const& fn : _labelFiles) { lfs.push_back(fn.c_str()); }
  }
  double* X = nullptr;
  int N, D;
  rf_old::readMatrixFromFiles(X, N, D, ffs);
  int* Y = nullptr;
  int Ny, Dy;
  rf_old::readMatrixFromFiles(Y, Ny, Dy, lfs);
  if (N != Ny || Dy != 1)
  { perr("Error: incorrect matrices dimension..."); }
  rf_old::TrainExtraOptions options;
  // Set sample size
  if (sampleSizeRatio > FEPS) {
    options.sampsize =
        rf_old::createIntScalar((int)((float)N * sampleSizeRatio + 0.5));
    options.n_sampsize = 1;
  }
  // Set classwt to balance class
  if (balance) {
    std::map<int, int> labelCount;
    rf_old::countLabel(labelCount, Y, N);
    int maxcnt = -1;
    for (auto cit = labelCount.begin(); cit != labelCount.end(); ++cit)
    { if (cit->second > maxcnt) { maxcnt = cit->second; } }
    options.classwt = new double[labelCount.size()];
    options.n_classwt = labelCount.size();
    int i = 0;
    for (auto cit = labelCount.begin(); cit != labelCount.end(); cit++) {
      options.classwt[i] = (double)maxcnt / (double)cit->second;
      std::cerr << "Class " << cit->first << ": " << cit->second
		<< " [w = " << options.classwt[i] << "]" << std::endl;
      i++;
    }
  }
  // Set node size
  options.nodesize = nodeSize;
  rf_old::Model model;
  rf_old::train(model, X, Y, N, D, options, nTree, mTry);
  rf_old::writeModelToBinaryFile(modelFile.c_str(), model);
  delete[] X;
  delete[] Y;
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("f", bpo::value<std::vector<std::string>>(&featFiles),
       "Input feature file name(s)")
      ("l", bpo::value<std::vector<std::string>>(&labelFiles),
       "Input label file name(s)")
      ("ff", bpo::value<std::string>(&featFileNameFile),
       "Input feature file name file name")
      ("ll", bpo::value<std::string>(&labelFileNameFile),
       "Input label file name file name")
      ("nt", bpo::value<int>(&nTree),
       "Number of trees for random forest [default: 255]")
      ("mt", bpo::value<int>(&mTry),
       "Number of features examined at each node for random forest "
       "(Use 0 for sqrt(D)) [default: 0]")
      ("sr", bpo::value<double>(&sampleSizeRatio),
       "Portion of samples used to train each tree for randomn forest "
       "[default: 0.7]")
      ("ns", bpo::value<int>(&nodeSize), "Node size [default: 1]")
      ("bal", bpo::value<bool>(&balance),
       "Whether to balance training by assigning weights [default: true]")
      ("m", bpo::value<std::string>(&modelFile)->required(),
       "Output model file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
