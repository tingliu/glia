#include "util/struct_merge.hxx"
#include "util/stats.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& segImageFile,
                std::string const& pbImageFile,
                std::string const& maskImageFile,
                std::vector<int> const& sizeThresholds,
                double rpbThreshold, bool relabel,
                bool write16, bool compress)
{
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto pbImage = readImage<RealImage<DIMENSION>>(pbImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, false);
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  std::unordered_map<Label, double> rpbs;
  genMergeOrderGreedyUsingPbMean
      (order, saliencies, rmap, true, pbImage,
       [&pbImage, &sizeThresholds, &rmap, rpbThreshold, &rpbs]
       (TBoundaryTable<std::pair<double, int>, RegionMap> const& bt,
        TBoundaryTable<std::pair<double, int>, RegionMap>::iterator btit)
       {
         Label key0 = btit->first.first;
         Label key1 = btit->first.second;
         auto const* pr0 = &rmap.find(key0)->second;
         auto const* pr1 = &rmap.find(key1)->second;
         auto sz0 = pr0->size();
         auto sz1 = pr1->size();
         if (sz0 > sz1) {
           std::swap(key0, key1);
           std::swap(pr0, pr1);
           std::swap(sz0, sz1);
         }
         if (sz0 < sizeThresholds[0]) { return true; }
         if (sizeThresholds.size() > 1) {
           if (sz0 < sizeThresholds[1]) {
             auto rpit0 = rpbs.find(key0);
             if (rpit0 != rpbs.end()) {
               if (rpit0->second > rpbThreshold) { return true; }
             } else {
               double rpb = 0.0;
               pr0->traverse(
                   [&pbImage, &rpb, rpbThreshold](
                       RegionMap::Region::Point const& p) {
                     rpb += pbImage->GetPixel(p); });
               rpb = sdivide(rpb, sz0, 0.0);
               rpbs[key0] = rpb;
               if (rpb > rpbThreshold) { return true; }
             }
           }
           if (sz1 < sizeThresholds[1]) {
             auto rpit1 = rpbs.find(key1);
             if (rpit1 != rpbs.end()) {
               if (rpit1->second > rpbThreshold) { return true; }
             } else {
               double rpb = 0.0;
               pr1->traverse(
                   [&pbImage, &rpb, rpbThreshold](
                       RegionMap::Region::Point const& p) {
                     rpb += pbImage->GetPixel(p); });
               rpb = sdivide(rpb, sz1, 0.0);
               rpbs[key1] = rpb;
               if (rpb > rpbThreshold) { return true; }
             }
           }
         }
         return false;
       });
  std::unordered_map<Label, Label> lmap;
  transformKeys(lmap, order);
  transformImage(segImage, rmap, lmap);
  if (relabel) { relabelImage(segImage, 0); }
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>
        (outputImageFile, segImage, compress);
  }
  else { writeImage(outputImageFile, segImage, compress); }
  return true;
}


int main (int argc, char* argv[])
{
  std::string segImageFile, pbImageFile, maskImageFile, outputImageFile;
  std::vector<int> sizeThresholds;
  double rpbThreshold;
  bool relabel = false, write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input image file name")
      ("pbImage,p", bpo::value<std::string>(&pbImageFile)->required(),
       "Input pb image file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("sizeThreshold,t", bpo::value<std::vector<int>>
       (&sizeThresholds)->required()->multitoken(),
       "Region size threshold(s) (e.g. -t 50 100)")
      ("rpbThreshold,b", bpo::value<double>(&rpbThreshold),
       "Region average boundary probability threshold")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel image to consecutive labels [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, segImageFile, pbImageFile, maskImageFile,
                sizeThresholds, rpbThreshold, relabel, write16, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
