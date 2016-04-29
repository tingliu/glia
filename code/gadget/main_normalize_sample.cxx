#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/stats.hxx"
#include "util/mp.hxx"
using namespace glia;

std::vector<std::string> inputFiles;
std::string inputMinMaxFile;
// FVal logThreshold = -1.0;  // Use log values if max is larger than this
FVal outputMin = -1.0;
FVal outputMax = 1.0;
std::vector<std::string> outputFiles;
std::string outputMinMaxFile;

bool operation ()
{
  int n = inputFiles.size();
  std::vector<std::vector<std::vector<FVal>>> feats(n);
  parfor(0, n, true, [&feats](int i) {
      readData(feats[i], inputFiles[i]); }, 0);
  // // Take log if necessary
  // if (logThreshold > 0.0) {
  //   std::vector<int> dims;
  //   for (auto& ffs : feats) {
  //     for (auto& ff : ffs) {
  //       int d = dims.empty() ? 0 : dims.back() + 1;
  //       for (auto& f : ff) {
  //         if (f >= logThreshold) {
  //           dims.push_back(d);
  //           break;
  //         }
  //         ++d;
  //       }
  //     }
  //   }
  //   for (auto& ffs : feats) {
  //     for (auto& ff : ffs)
  //     { for (int d : dims) { ff[d] = std::log(std::max(1.0, ff[d])); } }
  //   }
  // }
  std::vector<std::vector<FVal>> minmax;
  if (!inputMinMaxFile.empty()) { readData(minmax, inputMinMaxFile); }
  stats::rescale(minmax, feats, outputMin, outputMax);
  if (!outputFiles.empty()) {
    parfor (0, n, true, [&feats](int i) {
        writeData(outputFiles[i], feats[i], " ", "\n", FLT_PREC); }, 0);
  }
  if (inputMinMaxFile.empty() && !outputMinMaxFile.empty())
  { writeData(outputMinMaxFile, minmax, " ", "\n", FLT_PREC); }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("i", bpo::value<std::vector<std::string>>(&inputFiles)->required(),
       "Input file(s)")
      ("im", bpo::value<std::string>(&inputMinMaxFile),
       "Input min/max file")
      ("min", bpo::value<FVal>(&outputMin), "Output min [default: -1.0]")
      ("max", bpo::value<FVal>(&outputMax), "Output max [default: 1.0]")
      ("o", bpo::value<std::vector<std::string>>(&outputFiles),
       "Output file(s)")
      ("om", bpo::value<std::string>(&outputMinMaxFile),
       "Output min/max file");
  return parse(argc, argv, opts) &&
      operation() ? EXIT_SUCCESS : EXIT_FAILURE;
}
