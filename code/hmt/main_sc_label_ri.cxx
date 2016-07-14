#include "hmt/sc_label.hxx"
#include "type/big_num.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/mp.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;
using namespace glia::hmt;

std::string segImageFile0, segImageFile1;
std::string regionPairFile;
std::string truthImageFile0, truthImageFile1;
std::string maskImageFile0, maskImageFile1;
std::string scLabelFile;

bool operation ()
{
  typedef std::pair<int, Label> SRKey;
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  std::vector<std::pair<SRKey, SRKey>> regionPairs;
  readData(regionPairs, regionPairFile, true);
  auto segImage0 = readImage<LabelImage<DIMENSION>>(segImageFile0);
  auto segImage1 = readImage<LabelImage<DIMENSION>>(segImageFile1);
  auto truthImage0 = readImage<LabelImage<DIMENSION>>(truthImageFile0);
  auto truthImage1 = readImage<LabelImage<DIMENSION>>(truthImageFile1);
  auto mask0 = maskImageFile0.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile0);
  auto mask1 = maskImageFile1.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile1);
  RegionMap rmap0(segImage0, mask0, false), rmap1(segImage1, mask1, false);
  int n = regionPairs.size();
  std::vector<int> scLabels(n);
  parfor(0, n, true, [
      &scLabels, &rmap0, &rmap1, &regionPairs, &truthImage0, &truthImage1](
          int i) {
           auto const& rp = regionPairs[i];
           double trueF1, falseF1;
           scLabels[i] = genSectionClassificationLabelF1<BigInt>(
               trueF1, falseF1, rmap0.find(rp.first.second)->second,
               truthImage0, rmap1.find(rp.second.second)->second,
               truthImage1);
         }, 0);
  if (!scLabelFile.empty()) { writeData(scLabelFile, scLabels, "\n"); }
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
      ("rp", bpo::value<std::string>(&regionPairFile)->required(),
       "Input region pair file name")
      ("t0", bpo::value<std::string>(&truthImageFile0)->required(),
       "Input ground truth segmentation image file name 0")
      ("t1", bpo::value<std::string>(&truthImageFile1)->required(),
       "Input ground truth segmentation image file name 1")
      ("m0", bpo::value<std::string>(&maskImageFile0),
       "Input mask image file name 0 (optional)")
      ("m1", bpo::value<std::string>(&maskImageFile1),
       "Input mask image file name 1 (optional)")
      ("l", bpo::value<std::string>(&scLabelFile),
       "Output region pair label file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
