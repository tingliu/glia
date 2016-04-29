#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string segImageFile;
std::string maskImageFile;
int minSegSize = 0;
std::string truthImageFile;

bool operation ()
{
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto truthImage = readImage<LabelImage<DIMENSION>>(truthImageFile);
  auto mask = maskImageFile.empty() ? LabelImage<DIMENSION>::Pointer() :
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap segRegionMap(segImage, mask, false);
  RegionMap truthRegionMap(truthImage, mask, false);
  std::map<Label, std::pair<Label, int>> matchST;
  for (auto const& rp : segRegionMap) {
    std::unordered_map<Label, int> overlaps;
    stats::getOverlap(overlaps, rp.second, truthImage);
    if (overlaps.empty()) { continue; }
    int maxOverlap = 0;
    Label maxOverlapLabel = 0;
    for (auto const& op : overlaps) {
      if (op.second > maxOverlap) {
        maxOverlap = op.second;
        maxOverlapLabel = op.first;
      }
    }
    matchST[rp.first] = std::make_pair(maxOverlapLabel, maxOverlap);
  }
  std::map<Label, std::pair<
                    std::set<Label>, std::pair<double, double>>> matchTS;
  for (auto const& mstp : matchST) {
    int segRegionSize = segRegionMap.find(mstp.first)->second.size();
    if (segRegionSize >= minSegSize &&
        mstp.second.second > segRegionSize / 2.0) {
      auto mtsit = matchTS.find(mstp.second.first);
      if (mtsit == matchTS.end()) {
        auto& mtsp = matchTS[mstp.second.first];
        mtsp.first.insert(mstp.first);
        mtsp.second.first = mstp.second.second;
        mtsp.second.second = segRegionSize;
      } else {
        mtsit->second.first.insert(mstp.first);
        mtsit->second.second.first += mstp.second.second;
        mtsit->second.second.second += segRegionSize;
      }
    }
  }
  for (auto& mtsp : matchTS) {
    double overlapRatioT = mtsp.second.second.first /
        truthRegionMap.find(mtsp.first)->second.size();
    double overlapRatioS = mtsp.second.second.first /
        mtsp.second.second.second;
    double ji = mtsp.second.second.first /
        (truthRegionMap.find(mtsp.first)->second.size() +
         mtsp.second.second.second - mtsp.second.second.first);
    std::cout << mtsp.first << ": ";
    for (auto const& s : mtsp.second.first) {
      std::cout << s << " ("
                << segRegionMap.find(s)->second.size() << ") ";
    }
    std::cout << "[" << overlapRatioT << "] "
              << "[" << overlapRatioS << "] "
              << "[" << ji << "]" << std::endl;
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
      ("mins", bpo::value<int>(&minSegSize),
       "Mininum proposed segment size [default: 0]")
      ("t", bpo::value<std::string>(&truthImageFile)->required(),
       "Input truth image file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
