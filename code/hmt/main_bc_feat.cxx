#include "hmt/bc_feat.hxx"
#include "type/region_map.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
#include "hmt/hmt_util.hxx"
using namespace glia;
using namespace glia::hmt;

std::string segImageFile;
std::string mergeOrderFile;
std::string saliencyFile;
std::string pbImageFile;
std::vector<ImageFileHistPair> rbImageFiles;
std::vector<ImageFileHistPair> rlImageFiles;
std::vector<ImageFileHistPair> rImageFiles;
std::vector<ImageFileHistPair> bImageFiles;
std::string maskImageFile;
double initSal = 1.0;
double salBias = 1.0;
std::vector<double> boundaryThresholds;
bool normalizeShape = false;
bool useLogShape = false;
bool useSimpleFeatures = false;
std::string bfeatFile;

bool operation ()
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  // Load and set up images
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  RealImage<DIMENSION>::Pointer pbImage;
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
      rImages, rlImages, bImages;
  prepareImages(
      pbImage, rImages, rlImages, bImages, pbImageFile, rbImageFiles,
      rlImageFiles, rImageFiles, bImageFiles);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  // Set up normalizing area/length
  double normalizingArea =
      normalizeShape ? getImageVolume(segImage) : 1.0;
  double normalizingLength =
      normalizeShape ? getImageDiagonal(segImage) : 1.0;
  // Set up regions etc.
  std::vector<TTriple<Label>> order;
  readData(order, mergeOrderFile, true);
  std::vector<double> saliencies;
  std::unordered_map<Label, double> saliencyMap;
  if (!saliencyFile.empty()) {
    readData(saliencies, saliencyFile, true);
    genSaliencyMap(saliencyMap, order, saliencies, initSal, salBias);
  }
  RegionMap rmap(segImage, mask, order, false);
  // Generate region features
  int rn = rmap.size();
  std::vector<std::pair<Label, std::shared_ptr<RegionFeats>>> rfeats(rn);
  parfor(rmap, true, [
      &rfeats, &rImages, &rlImages, &bImages, &pbImage, &saliencyMap,
      normalizingArea, normalizingLength](
          RegionMap::const_iterator rit, int i) {
           rfeats[i].first = rit->first;
           rfeats[i].second = std::make_shared<RegionFeats>();
           rfeats[i].second->generate(
               rit->second, normalizingArea, normalizingLength,
               pbImage, boundaryThresholds, rImages, rlImages, bImages,
               ccpointer(saliencyMap, rit->first));
         }, 0);
  std::unordered_map<Label, std::shared_ptr<RegionFeats>> rfmap;
  for (auto const& rfp : rfeats) { rfmap[rfp.first] = rfp.second; }
  // Generate boundary classifier features
  if (!bfeatFile.empty()) {
    int bn = order.size();
    std::vector<BoundaryClassificationFeats> bfeats(bn);
    parfor(0, bn, true, [
        &rmap, &order, &bfeats, &rfmap, &bImages,
        &pbImage, normalizingLength](int i) {
             Label r0 = order[i].x0;
             Label r1 = order[i].x1;
             Label r2 = order[i].x2;
             bfeats[i].x1 = rfmap.find(r0)->second.get();
             bfeats[i].x2 = rfmap.find(r1)->second.get();
             bfeats[i].x3 = rfmap.find(r2)->second.get();
             // Keep region 0 area <= region 1 area
             if (bfeats[i].x1->shape->area > bfeats[i].x2->shape->area) {
               std::swap(r0, r1);
               std::swap(bfeats[i].x1, bfeats[i].x2);
             }
             RegionMap::Region::Boundary b;
             getBoundary(b, rmap.find(r0)->second, rmap.find(r1)->second);
             bfeats[i].x0.generate(
                 b, normalizingLength, *bfeats[i].x1, *bfeats[i].x2,
                 *bfeats[i].x3, pbImage, boundaryThresholds, bImages);
           }, 0);
    // Log shape
    if (useLogShape) {
      parfor(
          0, rn, false, [&rfeats](int i) { rfeats[i].second->log(); }, 0);
      parfor(
          0, bn, false, [&bfeats](int i) { bfeats[i].x0.log(); }, 0);
    }
    if (useSimpleFeatures) {
      std::vector<std::vector<FVal>> pickedFeats(bn);
      for (int i = 0; i < bn; ++i) {
        selectFeatures(pickedFeats[i], bfeats[i]);
      }
      writeData(bfeatFile, pickedFeats, " ", "\n", FLT_PREC);
    } else { writeData(bfeatFile, bfeats, "\n", FLT_PREC); }
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::vector<std::string>
      _rbImageFiles, _rlImageFiles, _rImageFiles, _bImageFiles;
  std::vector<unsigned int>
      _rbHistBins, _rlHistBins, _rHistBins, _bHistBins;
  std::vector<double>
      _rbHistLowers, _rlHistLowers, _rHistLowers, _bHistLowers;
  std::vector<double>
      _rbHistUppers, _rlHistUppers, _rHistUppers, _bHistUppers;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("saliency,y", bpo::value<std::string>(&saliencyFile),
       "Input merging saliency file name (optional)")
      ("rbi", bpo::value<std::vector<std::string>>(&_rbImageFiles),
       "Input real image file name(s) (optional)")
      ("rbb", bpo::value<std::vector<unsigned int>>(&_rbHistBins),
       "Input real image histogram bins")
      ("rbl", bpo::value<std::vector<double>>(&_rbHistLowers),
       "Input real image histogram lowers")
      ("rbu", bpo::value<std::vector<double>>(&_rbHistUppers),
       "Input real image histogram uppers")
      ("rli", bpo::value<std::vector<std::string>>(&_rlImageFiles),
       "Input region label image file names(s) (optional)")
      ("rlb", bpo::value<std::vector<unsigned int>>(&_rlHistBins),
       "Input region label image histogram bins")
      ("rll", bpo::value<std::vector<double>>(&_rlHistLowers),
       "Input region label image histogram lowers")
      ("rlu", bpo::value<std::vector<double>>(&_rlHistUppers),
       "Input region label image histogram uppers")
      ("ri", bpo::value<std::vector<std::string>>(&_rImageFiles),
       "Input excl. region image file name(s) (optional)")
      ("rb", bpo::value<std::vector<unsigned int>>(&_rHistBins),
       "Input excl. region image histogram bins")
      ("rl", bpo::value<std::vector<double>>(&_rHistLowers),
       "Input excl. region image histogram lowers")
      ("ru", bpo::value<std::vector<double>>(&_rHistUppers),
       "Input excl. boundary image histogram uppers")
      ("bi", bpo::value<std::vector<std::string>>(&_bImageFiles),
       "Input excl. boundary image file name(s) (optional)")
      ("bb", bpo::value<std::vector<unsigned int>>(&_bHistBins),
       "Input excl. boundary image histogram bins")
      ("bl", bpo::value<std::vector<double>>(&_bHistLowers),
       "Input excl. boundary image histogram lowers")
      ("bu", bpo::value<std::vector<double>>(&_bHistUppers),
       "Input excl. boundary image histogram uppers")
      ("pb", bpo::value<std::string>(&pbImageFile)->required(),
       "Boundary image file for image-based shape features")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name")
      ("s0", bpo::value<double>(&initSal),
       "Initial saliency [default: 1.0]")
      ("sb", bpo::value<double>(&salBias),
       "Saliency bias [default: 1.0]")
      ("bt",
       bpo::value<std::vector<double>>(&boundaryThresholds)->multitoken(),
       "Thresholds for image-based shape features (e.g. --bt 0.2 0.5 0.8)")
      ("ns,n", bpo::value<bool>(&normalizeShape),
       "Whether to normalize size and length [default: false]")
      ("logs,l", bpo::value<bool>(&useLogShape),
       "Whether to use logarithms of shape as features [default: false]")
      ("simpf", bpo::value<bool>(&useSimpleFeatures),
       "Whether to only use simplified features (following arXiv paper) "
       "[default: false]")
      ("bfeat,b", bpo::value<std::string>(&bfeatFile),
       "Output boundary feature file name (optional)");
  if (!parse(argc, argv, opts))
  { perr("Error: unable to parse input arguments"); }
  rbImageFiles.reserve(_rbImageFiles.size());
  rlImageFiles.reserve(_rlImageFiles.size());
  rImageFiles.reserve(_rImageFiles.size());
  bImageFiles.reserve(_bImageFiles.size());
  for (int i = 0; i < _rbImageFiles.size(); ++i) {
    rbImageFiles.emplace_back(_rbImageFiles[i], _rbHistBins[i],
                              _rbHistLowers[i], _rbHistUppers[i]);
  }
  for (int i = 0; i < _rlImageFiles.size(); ++i) {
    rlImageFiles.emplace_back(_rlImageFiles[i], _rlHistBins[i],
                              _rlHistLowers[i], _rlHistUppers[i]);
  }
  for (int i = 0; i < _rImageFiles.size(); ++i) {
    rImageFiles.emplace_back(_rImageFiles[i], _rHistBins[i],
                             _rHistLowers[i], _rHistUppers[i]);
  }
  for (int i = 0; i < _bImageFiles.size(); ++i) {
    bImageFiles.emplace_back(_bImageFiles[i], _bHistBins[i],
                             _bHistLowers[i], _bHistUppers[i]);
  }
  return operation() ? EXIT_SUCCESS: EXIT_FAILURE;
}
