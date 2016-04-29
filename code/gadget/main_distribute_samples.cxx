#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> inputFeatFiles;
std::vector<std::string> inputLabelFiles;
int featIndex0;
int featIndex1;
double threshold;
std::vector<std::string> outputFeatFiles;
std::vector<std::string> outputLabelFiles;

bool operation ()
{
  std::vector<std::vector<FVal>> feats;
  std::vector<int> labels;
  readData(feats, inputFeatFiles);
  readData(labels, inputLabelFiles, true);
  int n = feats.size();
  std::vector<std::vector<std::vector<FVal>>> dfeats(3);
  std::vector<std::vector<int>> dlabels(3);
  for (int i = 0; i < 3; ++i) {
    dfeats.reserve(n);
    dlabels.reserve(n);
  }
  for (int i = 0; i < n; ++i) {
    int ii = 2;
    if (feats[i][featIndex1] < threshold) { ii = 0; }
    else if (feats[i][featIndex0] < threshold) { ii = 1; }
    dfeats[ii].push_back(std::move(feats[i]));
    dlabels[ii].push_back(labels[i]);
  }
  for (int i = 0; i < 3; ++i) {
    writeData(outputFeatFiles[i], dfeats[i], " ", "\n", FLT_PREC);
    writeData(outputLabelFiles[i], dlabels[i], "\n");
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("f", bpo::value<std::vector<std::string>>(
          &inputFeatFiles)->required(), "Input feature file name(s)")
      ("l", bpo::value<std::vector<std::string>>(
          &inputLabelFiles)->required(), "Input label file name(s)")
      ("i0", bpo::value<int>(&featIndex0)->required(),
       "Input smaller-valued feature index")
      ("i1", bpo::value<int>(&featIndex1)->required(),
       "Input larger-valued feature index")
      ("t", bpo::value<double>(&threshold)->required(), "Threshold")
      ("of", bpo::value<std::vector<std::string>>(
          &outputFeatFiles)->required(),
       "Output distributed feature file name(s)")
      ("ol", bpo::value<std::vector<std::string>>(
          &outputLabelFiles)->required(),
       "Output distributed label file name(s)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
