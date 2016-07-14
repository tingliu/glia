#include "type/region_map.hxx"
#include "alg/geometry.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

std::string segImageFile0, segImageFile1;
std::string maskImageFile0, maskImageFile1;
int imageId0, imageId1;
double maxCentroidDist = -1.0;
std::string regionPairFile;

bool operation ()
{
  typedef std::pair<int, Label> SRKey;
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  typedef fPoint<RegionMap::Point::Dimension> CPoint;
  auto segImage0 = readImage<LabelImage<DIMENSION>>(segImageFile0);
  auto segImage1 = readImage<LabelImage<DIMENSION>>(segImageFile1);
  auto mask0 = maskImageFile0.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile0);
  auto mask1 = maskImageFile1.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile1);
  RegionMap rmap0(segImage0, mask0, false), rmap1(segImage1, mask1, false);
  std::unordered_map<std::pair<Label, Label>, int> overlaps;
  stats::getOverlap(
      overlaps, segImage0, mask0, {BG_VAL}, segImage1, mask1, {BG_VAL});
  std::map<Label, CPoint> centroids0, centroids1;
  for (auto const& rp : rmap0) { centroids0[rp.first]; }
  for (auto const& rp : rmap1) { centroids1[rp.first]; }
  parfor(centroids0, true, [&rmap0](
      std::map<Label, CPoint>::iterator cit, int i) {
           alg::getCentroid(cit->second, rmap0.find(cit->first)->second);
         }, 0);
  parfor(centroids1, true, [&rmap1](
      std::map<Label, CPoint>::iterator cit, int i) {
           alg::getCentroid(cit->second, rmap1.find(cit->first)->second);
         }, 0);
  std::vector<std::pair<SRKey, SRKey>> regionPairs;
  for (auto const& cp0 : centroids0) {
    for (auto const& cp1 : centroids1) {
      if (overlaps.count(std::make_pair(cp0.first, cp1.first)) > 0 ||
          cp0.second.EuclideanDistanceTo(cp1.second) <= maxCentroidDist) {
        regionPairs.emplace_back(std::make_pair(
            std::make_pair(imageId0, cp0.first),
            std::make_pair(imageId1, cp1.first)));
      }
    }
  }
  writeData(regionPairFile, regionPairs, "\n");
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("s0", bpo::value<std::string>(&segImageFile0)->required(),
       "Input segmentation image file name 0")
      ("s1", bpo::value<std::string>(&segImageFile1)->required(),
       "Input segmentation image file name 1")
      ("m0", bpo::value<std::string>(&maskImageFile0),
       "Input mask image file name 0")
      ("m1", bpo::value<std::string>(&maskImageFile1),
       "Input mask image file name 1")
      ("id0", bpo::value<int>(&imageId0)->required(), "Image ID 0")
      ("id1", bpo::value<int>(&imageId1)->required(), "Image ID 1")
      ("cd", bpo::value<double>(&maxCentroidDist),
       "Max centroid distance (Use -1 to enforce overlap) [default: -1]")
      ("rp", bpo::value<std::string>(&regionPairFile)->required(),
       "Output region pair file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
