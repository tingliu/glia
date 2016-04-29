#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> inputFeatFiles;
std::vector<std::string> inputLabelFiles;
std::vector<std::string> inputPredFiles;
int label0 = 1;
int label1 = -1;
double threshold0 = 0.5;
double threshold1 = 0.5;
std::string outputFeatFile;
std::string outputLabelFile;

bool operation ()
{
  std::vector<std::vector<FVal>> feats;
  std::vector<int> labels;
  std::vector<double> preds;
  readData(feats, inputFeatFiles);
  readData(labels, inputLabelFiles, true);
  readData(preds, inputPredFiles, true);
  int n = feats.size();
  std::vector<std::vector<FVal>> outputFeats;
  std::vector<int> outputLabels;
  outputFeats.reserve(n);
  outputLabels.reserve(n);
  for (int i = 0; i < n; ++i) {
    // Only select samples that are on incorrect side of decision plane
    if (labels[i] == label0 && preds[i] > threshold0 ||
        labels[i] == label1 && preds[i] < threshold1) {
      outputFeats.push_back(std::move(feats[i]));
      outputLabels.push_back(labels[i]);
    }
  }
  writeData(outputFeatFile, outputFeats, " ", "\n", FLT_PREC);
  writeData(outputLabelFile, outputLabels, "\n");
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
      ("p", bpo::value<std::vector<std::string>>(
          &inputPredFiles)->required(), "Input prediction file name(s)")
      ("l0", bpo::value<int>(&label0),
       "Label for prediction 0 [default: 1 for hmt/bc]")
      ("l1", bpo::value<int>(&label1),
       "Label for prediction 1 [default: -1 for hmt/bc]")
      ("t0", bpo::value<double>(&threshold0),
       "Decision threshold for prediction 0 [default: 0.5]")
      ("t1", bpo::value<double>(&threshold1),
       "Decision threshold for prediction 1 [default: 0.5]")
      ("of", bpo::value<std::string>(
          &outputFeatFile)->required(), "Output feature file name")
      ("ol", bpo::value<std::string>(
          &outputLabelFile)->required(), "Output label file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
