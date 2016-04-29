#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string segImageFile;
std::string maskImageFile;
std::string truthImageFile;

bool operation ()
{
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto truthImage = readImage<LabelImage<DIMENSION>>(truthImageFile);
  auto mask = maskImageFile.empty() ? LabelImage<DIMENSION>::Pointer() :
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  std::unordered_map<Label, int> segSizeMap;
  genCountMap(segSizeMap, segImage, mask);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap truthRegionMap(truthImage, mask, false);
  std::map<Label, std::pair<Label, double>> matchTS;
  for (auto const& rp : truthRegionMap) {
    if (rp.first != BG_VAL) {
      std::unordered_map<Label, int> overlaps;
      stats::getOverlap(overlaps, rp.second, segImage);
      if (overlaps.empty()) { continue; }
      double maxJI = 0.0;
      Label maxJiLabel = 0;
      for (auto const& op : overlaps) {
        double ji = (double)op.second /
            (segSizeMap.find(op.first)->second + rp.second.size()
             - op.second);
        if (ji > maxJI) {
          maxJI = ji;
          maxJiLabel = op.first;
        }
      }
      matchTS[rp.first] = std::make_pair(maxJiLabel, maxJI);
    }
  }
  for (auto const& mp : matchTS) {
    std::cout << mp.first << ": " << mp.second.first << " ["
              << mp.second.second << "]" << std::endl;
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("s", bpo::value<std::string>(&segImageFile)->required(),
       "Input segmentation image file name")
      ("m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name")
      ("t", bpo::value<std::string>(&truthImageFile)->required(),
       "Input truth image file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
