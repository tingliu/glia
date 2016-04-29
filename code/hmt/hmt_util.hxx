#include "util/image_io.hxx"
using namespace glia;
using namespace glia::hmt;

struct ImageFileHistPair {
  std::string imageFile;
  unsigned int histBin;
  std::pair<double, double> histRange;

  ImageFileHistPair (std::string const& imageFile, unsigned int histBin,
                     double histLower, double histUpper)
      : imageFile(imageFile), histBin(histBin)
  { histRange = std::make_pair(histLower, histUpper); }
};


void prepareImages (
    RealImage<DIMENSION>::Pointer& pbImage,
    std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>& rImages,
    std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>& rlImages,
    std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>& bImages,
    std::string const& pbImageFile,
    std::vector<ImageFileHistPair> const& rbImageFiles,
    std::vector<ImageFileHistPair> const& rlImageFiles,
    std::vector<ImageFileHistPair> const& rImageFiles,
    std::vector<ImageFileHistPair> const& bImageFiles)
{
  rImages.reserve(rbImageFiles.size() + rImageFiles.size());
  rlImages.reserve(rlImageFiles.size());
  bImages.reserve(bImageFiles.size() + rbImageFiles.size());
  for (auto const& ifhp: rbImageFiles) {
    rImages.emplace_back(
        readImage<RealImage<DIMENSION>>(
            ifhp.imageFile), ifhp.histBin, ifhp.histRange);
    bImages.push_back(rImages.back());
    if (pbImageFile == ifhp.imageFile) { pbImage = bImages.back().image; }
  }
  for (auto const& ifhp: rImageFiles) {
    rImages.emplace_back(
        readImage<RealImage<DIMENSION>>(
            ifhp.imageFile), ifhp.histBin, ifhp.histRange);
  }
  for (auto const& ifhp: bImageFiles) {
    bImages.emplace_back(
        readImage<RealImage<DIMENSION>>(
            ifhp.imageFile), ifhp.histBin, ifhp.histRange);
    if (pbImageFile == ifhp.imageFile) { pbImage = bImages.back().image; }
  }
  for (auto const& ifhp: rlImageFiles) {
    rlImages.emplace_back(
        readImage<RealImage<DIMENSION>>(
            ifhp.imageFile), ifhp.histBin, ifhp.histRange);
  }
  if (pbImage.IsNull())
  { pbImage = readImage<RealImage<DIMENSION>>(pbImageFile); }
}
