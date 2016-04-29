#include "util/text_io.hxx"
#include "util/container.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

struct Sample {
  int label;
  std::vector<FVal>* pFeat;

  Sample (int label, std::vector<FVal>* pFeat)
      : label(label), pFeat(pFeat) {}

  bool operator < (Sample const& x) const
  { return label < x.label || (label == x.label && *pFeat < *x.pFeat); }

  bool operator == (Sample const& x) const
  { return label == x.label && *pFeat == *x.pFeat; }
};

std::vector<std::string> featFiles;
std::vector<std::string> labelFiles;
std::string uniqueFeatFile;
std::string uniqueLabelFile;

bool operation ()
{
  std::vector<std::vector<FVal>> feats, ufeats;
  std::vector<int> labels, ulabels;
  readData(feats, featFiles);
  readData(labels, labelFiles, true);
  std::set<Sample> samples;
  int n = feats.size();
  for (int i = 0; i < n; ++i)
  { samples.emplace(Sample(labels[i], &feats[i])); }
  ufeats.reserve(samples.size());
  ulabels.reserve(samples.size());
  for (auto const& x: samples) {
    ulabels.push_back(x.label);
    ufeats.push_back(std::vector<FVal>());
    splice(ufeats.back(), *x.pFeat);
  }
  writeData(uniqueFeatFile, ufeats, " ", "\n", FLT_PREC);
  writeData(uniqueLabelFile, ulabels, "\n", FLT_PREC);
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("feat,f",
       bpo::value<std::vector<std::string>>(&featFiles)->required(),
       "Input feature file name(s)")
      ("label,l",
       bpo::value<std::vector<std::string>>(&labelFiles)->required(),
       "Input label file name(s)")
      ("ufeat,u", bpo::value<std::string>(&uniqueFeatFile)->required(),
       "Output unique feature file name")
      ("ulabel,o", bpo::value<std::string>(&uniqueLabelFile)->required(),
       "Output unique label file name");
  if (!parse(argc, argv, opts)) { return EXIT_FAILURE; }
  if (featFiles.size() != labelFiles.size())
  { perr("Error: feature/label file numbers disagree..."); }
  return operation()? EXIT_SUCCESS: EXIT_FAILURE;
}
