#include "hmt/bc_feat.hxx"
#include "hmt/bc_label.hxx"
#include "alg/rf.hxx"
#include "alg/nn.hxx"
#include "util/struct_merge_bc.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "hmt/hmt_util.hxx"
using namespace glia;
using namespace glia::hmt;

int bcType;  // 1: random forest, 2: MLP2
int nNodeLayer1 = 10;  // Only used if bcType == 2
int nNodeLayer2 = 10;  // Only used if bcType == 2
std::vector<std::string> bcModelFiles;
std::string bcFeatMinMaxFile;  // Only used if features need normalizing
std::vector<double> bcModelDistributorArgs;
std::string segImageFile;
std::string maskImageFile;
std::string pbImageFile;
std::vector<ImageFileHistPair> rbImageFiles;
std::vector<ImageFileHistPair> rlImageFiles;
std::vector<ImageFileHistPair> rImageFiles;
std::vector<ImageFileHistPair> bImageFiles;
std::vector<double> boundaryThresholds;
bool normalizeShape = false;
bool useLogShape = false;
bool useSimpleFeatures = false;
std::string mergeOrderFile;
std::string saliencyFile;
std::string bcFeatFile;

bool operation ()
{
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  double normalizingArea =
      normalizeShape ? getImageVolume(segImage) : 1.0;
  double normalizingLength =
      normalizeShape ? getImageDiagonal(segImage) : 1.0;
  auto mask = maskImageFile.empty() ?
      LabelImage<DIMENSION>::Pointer(nullptr) :
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  RealImage<DIMENSION>::Pointer pbImage;
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
      rImages, rlImages, bImages;
  prepareImages(
      pbImage, rImages, rlImages, bImages, pbImageFile, rbImageFiles,
      rlImageFiles, rImageFiles, bImageFiles);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, false);
  // Boundary feature computer
  std::unordered_map<std::pair<Label, Label>, std::vector<FVal>>
      bcfmap;
  auto fBcFeat = [
      normalizingArea, normalizingLength, &pbImage, &rImages,
      &rlImages, &bImages, &bcfmap](
          std::vector<FVal>& data, RegionMap::Region const& reg0,
          RegionMap::Region const& reg1, RegionMap::Region const& reg2,
          Label r0, Label r1, Label r2) {
    auto rf0 = std::make_shared<RegionFeats>();
    auto rf1 = std::make_shared<RegionFeats>();
    auto rf2 = std::make_shared<RegionFeats>();
    rf0->generate(
        reg0, normalizingArea, normalizingLength, pbImage,
        boundaryThresholds, rImages, rlImages, bImages, nullptr);
    rf1->generate(
        reg1, normalizingArea, normalizingLength, pbImage,
        boundaryThresholds, rImages, rlImages, bImages, nullptr);
    rf2->generate(
        reg2, normalizingArea, normalizingLength, pbImage,
        boundaryThresholds, rImages, rlImages, bImages, nullptr);
    BoundaryClassificationFeats bcf;
    bcf.x1 = rf0.get();
    bcf.x2 = rf1.get();
    bcf.x3 = rf2.get();
    // Keep region 0 area <= region 1 area
    if (bcf.x1->shape->area > bcf.x2->shape->area) {
      std::swap(r0, r1);
      std::swap(bcf.x1, bcf.x2);
    }
    RegionMap::Region::Boundary b;
    getBoundary(b, reg0, reg1);
    bcf.x0.generate(
        b, normalizingLength, *bcf.x1, *bcf.x2, *bcf.x3, pbImage,
        boundaryThresholds, bImages);
    if (useLogShape) {
      bcf.x0.log();
      rf0->log();
      rf1->log();
      rf2->log();
    }
    if (useSimpleFeatures) { selectFeatures(data, bcf); }
    else { bcf.serialize(data); }
    bcfmap[std::make_pair(std::min(r0, r1), std::max(r0, r1))] = data;
  };
  std::shared_ptr<opt::TFunction<std::vector<FVal>>> bc;
  std::vector<std::vector<FVal>> bcfMinMax;
  if (bcType == 1) {  // Random forest
    if (bcModelFiles.size() == 1) {
      bc = std::make_shared<alg::RandomForest>(
          BC_LABEL_MERGE, bcModelFiles.front());
    } else {
      if (bcModelDistributorArgs.size() != 3)
      { perr("Error: model distributor needs 3 arguments..."); }
      bc = std::make_shared<alg::EnsembleRandomForest>(
          BC_LABEL_MERGE, bcModelFiles,
          opt::ThresholdModelDistributor<FVal>(
              bcModelDistributorArgs[0], bcModelDistributorArgs[1],
              bcModelDistributorArgs[2]));
    }
  } else if (bcType == 2) {  // MLP2
    readData(bcfMinMax, bcFeatMinMaxFile);
    if (bcModelFiles.size() == 1) {
      bc = std::make_shared<alg::MLP2v>(
          nNodeLayer1, nNodeLayer2, bcModelFiles.front());
    } else {
      if (bcModelDistributorArgs.size() != 3)
      { perr("Error: model distributor needs 3 arguments..."); }
      bc = std::make_shared<alg::EnsembleMLP2v>(
          nNodeLayer1, nNodeLayer2, bcModelFiles,
          opt::ThresholdModelDistributor<FVal>(
              bcModelDistributorArgs[0], bcModelDistributorArgs[1],
              bcModelDistributorArgs[2]));
    }
  } else { perr("Error: unsupported classifier type..."); }
  // Boundary predictor
  std::vector<FVal> tmpData;
  auto fBcPred = [&bc, &bcfMinMax, &tmpData](
      std::vector<FVal> const& data) {
    if (bcType == 2) {  // MLP
      tmpData.clear();
      tmpData.reserve(data.size() + 1);
      tmpData.insert(tmpData.end(), data.begin(), data.end());
      stats::rescale(tmpData, bcfMinMax, -1.0, 1.0);
      tmpData.push_back(1.0);  //  Do not forget bias!
      return bc->operator()(tmpData);
    }
    return bc->operator()(data);
  };
  // Generate merging orders
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  genMergeOrderGreedyUsingBoundaryClassifier<std::vector<FVal>>(
      order, saliencies, segImage, mask, fBcFeat, fBcPred,
      f_true<TBoundaryTable<std::vector<FVal>, RegionMap>&,
      TBoundaryTable<std::vector<FVal>, RegionMap>::iterator>);
  // Output
  if (!mergeOrderFile.empty()) { writeData(mergeOrderFile, order, "\n"); }
  if (!saliencyFile.empty()) { writeData(saliencyFile, saliencies, "\n"); }
  if (!bcFeatFile.empty()) {
    std::vector<std::vector<FVal>> bcfeats;
    bcfeats.reserve(order.size());
    for (auto const& m : order) {
      bcfeats.push_back(bcfmap.find(std::make_pair(m.x0, m.x1))->second);
    }
    writeData(bcFeatFile, bcfeats, " ", "\n", FLT_PREC);
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
      ("bct", bpo::value<int>(&bcType)->required(),
       "Boundary classifier type (1: random forest, 2: MLP2)")
      ("nn1", bpo::value<int>(&nNodeLayer1),
       "Number of nodes on hidden layer 1 for MLP2 [default: 10]")
      ("nn2", bpo::value<int>(&nNodeLayer2),
       "Number of nodes on hidden layer 2 for MLP2 [default: 10]")
      ("bcm",
       bpo::value<std::vector<std::string>>(&bcModelFiles)->required(),
       "Input boundary classification model file name(s)")
      ("bcfmm", bpo::value<std::string>(&bcFeatMinMaxFile),
       "Input feature min/max file name")
      ("bcmd", bpo::value<std::vector<double>>(&bcModelDistributorArgs),
       "Boundary classifier distributor argument(s) "
       "(e.g. --bcmd 0 --bcmd 1 --bcmd areaMedian)")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
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
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Output merging order file name")
      ("sal", bpo::value<std::string>(&saliencyFile),
       "Output merging saliency file name (optional)")
      ("bfeat,b", bpo::value<std::string>(&bcFeatFile),
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
