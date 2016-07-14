#include "hmt/sc_feat.hxx"
#include "hmt/hmt_util.hxx"
#include "type/region_map.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

std::string segImageFile0, segImageFile1;
std::string regionPairFile;
std::string pbImageFile0, pbImageFile1;
std::vector<ImageFileHistPair> rbImageFiles0, rbImageFiles1;
std::vector<ImageFileHistPair> rlImageFiles0, rlImageFiles1;
std::vector<ImageFileHistPair> rImageFiles0, rImageFiles1;
std::vector<ImageFileHistPair> bImageFiles0, bImageFiles1;
std::string maskImageFile0, maskImageFile1;
std::vector<double> boundaryThresholds;
double normalizingArea = 1.0;
double normalizingLength = 1.0;
bool useLogShape = false;
std::string sfeatFile;

bool operation ()
{
  typedef std::pair<int, Label> SRKey;
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  auto segImage0 = readImage<LabelImage<DIMENSION>>(segImageFile0);
  auto segImage1 = readImage<LabelImage<DIMENSION>>(segImageFile1);
  std::vector<std::pair<SRKey, SRKey>> regionPairs;
  readData(regionPairs, regionPairFile, true);
  RealImage<DIMENSION>::Pointer pbImage0, pbImage1;
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
      rImages0, rlImages0, bImages0, rImages1, rlImages1, bImages1;
  prepareImages(
      pbImage0, rImages0, rlImages0, bImages0, pbImageFile0, rbImageFiles0,
      rlImageFiles0, rImageFiles0, bImageFiles0);
  prepareImages(
      pbImage1, rImages1, rlImages1, bImages1, pbImageFile1, rbImageFiles1,
      rlImageFiles1, rImageFiles1, bImageFiles1);
  auto mask0 = maskImageFile0.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile0);
  auto mask1 = maskImageFile1.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile1);
  // Set up normalizing area/length
  if (normalizingArea <= 0.0) {
    normalizingArea = std::max(
        getImageVolume(segImage0), getImageVolume(segImage1));
  }
  if (normalizingLength <= 0.0) {
    normalizingLength = std::max(
        getImageDiagonal(segImage0), getImageDiagonal(segImage1));
  }
  // Set up regions etc.
  RegionMap rmap0(segImage0, mask0, false), rmap1(segImage1, mask1, false);
  // Generate region features;
  std::vector<std::pair<Label, std::shared_ptr<RegionFeatsWithLocation>>>
      rfeats0(rmap0.size()), rfeats1(rmap1.size());
  parfor(rmap0, true, [
      &rfeats0, &rImages0, &rlImages0, &bImages0, &pbImage0](
          RegionMap::const_iterator rit, int i) {
           rfeats0[i].first = rit->first;
           rfeats0[i].second =
               std::make_shared<RegionFeatsWithLocation>();
           rfeats0[i].second->generate(
               rit->second, normalizingArea, normalizingLength,
               pbImage0, boundaryThresholds, rImages0, rlImages0, bImages0,
               nullptr);
         }, 0);
  parfor(rmap1, true, [
      &rfeats1, &rImages1, &rlImages1, &bImages1, &pbImage1](
          RegionMap::const_iterator rit, int i) {
           rfeats1[i].first = rit->first;
           rfeats1[i].second =
               std::make_shared<RegionFeatsWithLocation>();
           rfeats1[i].second->generate(
               rit->second, normalizingArea, normalizingLength,
               pbImage1, boundaryThresholds, rImages1, rlImages1, bImages1,
               nullptr);
         }, 0);
  std::unordered_map<Label, std::shared_ptr<RegionFeatsWithLocation>>
      rfmap0, rfmap1;
  for (auto const& rfp : rfeats0) { rfmap0[rfp.first] = rfp.second; }
  for (auto const& rfp : rfeats1) { rfmap1[rfp.first] = rfp.second; }
  // Generate region overlaps
  std::unordered_map<std::pair<Label, Label>, int> overlaps;
  stats::getOverlap(
      overlaps, segImage0, mask0, {BG_VAL}, segImage1, mask1, {BG_VAL});
  // Generate section classifier features
  if (!sfeatFile.empty()) {
    int sn = regionPairs.size();
    std::vector<SectionClassificationFeats> sfeats(sn);
    parfor(0, sn, true, [
        &regionPairs, &sfeats, &rfmap0, &rfmap1, &overlaps](int i) {
             Label r0 = regionPairs[i].first.second;
             Label r1 = regionPairs[i].second.second;
             auto rf0 = rfmap0.find(r0)->second.get();
             auto rf1 = rfmap1.find(r1)->second.get();
             sfeats[i].x0.generate(
                 *rf0, *rf1, (double)clookup(
                     overlaps, std::make_pair(r0, r1), 0));
             // Keep region 0 area <= region 1 area
             if (rf0->shape->area > rf1->shape->area) {
               std::swap(r0, r1);
               std::swap(rf0, rf1);
             }
             sfeats[i].x1 = rf0;
             sfeats[i].x2 = rf1;
           }, 0);
    if (useLogShape) {
      parfor(0, rfeats0.size(), false, [&rfeats0](int i) {
          rfeats0[i].second->log(); }, 0);
      parfor(0, rfeats1.size(), false, [&rfeats1](int i) {
          rfeats1[i].second->log(); }, 0);
      parfor(0, sn, false, [&sfeats](int i) {
          sfeats[i].x0.log(); }, 0);
    }
    writeData(sfeatFile, sfeats, "\n", FLT_PREC);
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::vector<std::string>
      _rbImageFiles0, _rlImageFiles0, _rImageFiles0, _bImageFiles0,
      _rbImageFiles1, _rlImageFiles1, _rImageFiles1, _bImageFiles1;
  std::vector<unsigned int>
      _rbHistBins, _rlHistBins, _rHistBins, _bHistBins;
  std::vector<double>
      _rbHistLowers, _rlHistLowers, _rHistLowers, _bHistLowers;
  std::vector<double>
      _rbHistUppers, _rlHistUppers, _rHistUppers, _bHistUppers;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("s0", bpo::value<std::string>(&segImageFile0)->required(),
       "Input segmentation image file name 0")
      ("s1", bpo::value<std::string>(&segImageFile1)->required(),
       "Input segmentation image file name 1")
      ("rp", bpo::value<std::string>(&regionPairFile)->required(),
       "Input region pair file name")
      ("rbi0", bpo::value<std::vector<std::string>>(&_rbImageFiles0),
       "Input real image file name(s) 0 (optional)")
      ("rbi1", bpo::value<std::vector<std::string>>(&_rbImageFiles1),
       "Input real image file name(s) 1 (optional)")
      ("rbb", bpo::value<std::vector<unsigned int>>(&_rbHistBins),
       "Input real image histogram bins")
      ("rbl", bpo::value<std::vector<double>>(&_rbHistLowers),
       "Input real image histogram lowers")
      ("rbu", bpo::value<std::vector<double>>(&_rbHistUppers),
       "Input real image histogram uppers")
      ("rli0", bpo::value<std::vector<std::string>>(&_rlImageFiles0),
       "Input region label image file name(s) 0 (optional)")
      ("rli1", bpo::value<std::vector<std::string>>(&_rlImageFiles1),
       "Input region label image file name(s) 1 (optional)")
      ("rlb", bpo::value<std::vector<unsigned int>>(&_rlHistBins),
       "Input region label image histogram bins")
      ("rll", bpo::value<std::vector<double>>(&_rlHistLowers),
       "Input region label image histogram lowers")
      ("rlu", bpo::value<std::vector<double>>(&_rlHistUppers),
       "Input region label image histogram uppers")
      ("ri0", bpo::value<std::vector<std::string>>(&_rImageFiles0),
       "Input excl. region image file name(s) 0 (optional)")
      ("ri1", bpo::value<std::vector<std::string>>(&_rImageFiles1),
       "Input excl. region image file name(s) 1 (optional)")
      ("rb", bpo::value<std::vector<unsigned int>>(&_rHistBins),
       "Input excl.  region image histogram bins")
      ("rl", bpo::value<std::vector<double>>(&_rHistLowers),
       "Input excl. region image histogram lowers")
      ("ru", bpo::value<std::vector<double>>(&_rHistUppers),
       "Input excl. region image histogram uppers")
      ("bi0", bpo::value<std::vector<std::string>>(&_bImageFiles0),
       "Input excl. boundary image file name(s) 0 (optional)")
      ("bi1", bpo::value<std::vector<std::string>>(&_bImageFiles1),
       "Input excl. boundary image file name(s) 1 (optional)")
      ("bb", bpo::value<std::vector<unsigned int>>(&_bHistBins),
       "Input excl.  boundary image histogram bins")
      ("bl", bpo::value<std::vector<double>>(&_bHistLowers),
       "Input excl. boundary image histogram lowers")
      ("bu", bpo::value<std::vector<double>>(&_bHistUppers),
       "Input excl. boundary image histogram uppers")
      ("pb0", bpo::value<std::string>(&pbImageFile0)->required(),
       "Boundary image file 0 for image-based shape features")
      ("pb1", bpo::value<std::string>(&pbImageFile1)->required(),
       "Boundary image file 1 for image-based shape features")
      ("m0", bpo::value<std::string>(&maskImageFile0),
       "Input mask image file name 0")
      ("m1", bpo::value<std::string>(&maskImageFile1),
       "Input mask image file name 1")
      ("bt",
       bpo::value<std::vector<double>>(&boundaryThresholds)->multitoken(),
       "Thresholds for image-based shape features (e.g. --bt 0.2 0.5 0.8)")
      ("na", bpo::value<double>(&normalizingArea),
       "Normalizing area (Use 1.0 to bypass; "
       "-1 to use max image volume) [default: 1.0]")
      ("nl", bpo::value<double>(&normalizingLength),
       "Normalizing length (Use 1.0 to bypass; "
       "-1 to use max image diagonal) [default: 1.0]")
      ("logs,l", bpo::value<bool>(&useLogShape),
       "Whether to use logarithms of shape as features [default: false]")
      ("f", bpo::value<std::string>(&sfeatFile),
       "Output section feature file name (optional)");
  if (!parse(argc, argv, opts))
  { perr("Error: unable to parse input arguments"); }
  rbImageFiles0.reserve(_rbImageFiles0.size());
  rlImageFiles0.reserve(_rlImageFiles0.size());
  rImageFiles0.reserve(_rImageFiles0.size());
  bImageFiles0.reserve(_bImageFiles0.size());
  rbImageFiles1.reserve(_rbImageFiles1.size());
  rlImageFiles1.reserve(_rlImageFiles1.size());
  rImageFiles1.reserve(_rImageFiles1.size());
  bImageFiles1.reserve(_bImageFiles1.size());
  for (int i = 0; i < _rbImageFiles0.size(); ++i) {
    rbImageFiles0.emplace_back(_rbImageFiles0[i], _rbHistBins[i],
                               _rbHistLowers[i], _rbHistUppers[i]);
    rbImageFiles1.emplace_back(_rbImageFiles1[i], _rbHistBins[i],
                               _rbHistLowers[i], _rbHistUppers[i]);
  }
  for (int i = 0; i < _rlImageFiles0.size(); ++i) {
    rlImageFiles0.emplace_back(_rlImageFiles0[i], _rlHistBins[i],
                               _rlHistLowers[i], _rlHistUppers[i]);
    rlImageFiles1.emplace_back(_rlImageFiles1[i], _rlHistBins[i],
                               _rlHistLowers[i], _rlHistUppers[i]);
  }
  for (int i = 0; i < _rImageFiles0.size(); ++i) {
    rImageFiles0.emplace_back(_rImageFiles0[i], _rHistBins[i],
                              _rHistLowers[i], _rHistUppers[i]);
    rImageFiles1.emplace_back(_rImageFiles1[i], _rHistBins[i],
                              _rHistLowers[i], _rHistUppers[i]);
  }
  for (int i = 0; i < _bImageFiles0.size(); ++i) {
    bImageFiles0.emplace_back(_bImageFiles0[i], _bHistBins[i],
                              _bHistLowers[i], _bHistUppers[i]);
    bImageFiles1.emplace_back(_bImageFiles1[i], _bHistBins[i],
                              _bHistLowers[i], _bHistUppers[i]);
  }
  return operation() ? EXIT_SUCCESS: EXIT_FAILURE;
}
