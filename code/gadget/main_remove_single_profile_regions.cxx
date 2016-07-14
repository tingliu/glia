#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

std::vector<std::string> inputImageFiles;
std::vector<std::string> maskImageFiles;
Label outputBgVal = BG_VAL;
bool relabel = false;
bool write16 = false;
bool compress = false;
std::vector<std::string> outputImageFiles;

bool operation ()
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  std::vector<LabelImage<DIMENSION>::Pointer> images;
  std::vector<LabelImage<DIMENSION>::Pointer> masks;
  // Read images
  int n = inputImageFiles.size();
  if (n == 1) {  // Input is volume
    auto vol = readImage<LabelImage<DIMENSION + 1>>(
        inputImageFiles.front());
    n = getImageSize(vol, DIMENSION);
    images.resize(n);
    std::vector<int> size(DIMENSION + 1);
    for (int i = 0; i < DIMENSION; ++i) { size[i] = getImageSize(vol, i); }
    size[DIMENSION] = 0;
    for (int i = 0; i < n; ++i) {
      std::vector<int> start(DIMENSION + 1, 0);
      start[DIMENSION] = i;
      images[i] = extractImage<DIMENSION>(vol, start, size);
    }
  } else {  // Inputs are slices
    images.resize(n);
    for (int i = 0; i < n; ++i)
    { images[i] = readImage<LabelImage<DIMENSION>>(inputImageFiles[i]); }
  }
  // Read masks
  masks.resize(n);
  if (maskImageFiles.size() == 1) {
    auto maskVol = readImage<LabelImage<DIMENSION + 1>>(
        maskImageFiles.front());
    std::vector<int> size(DIMENSION + 1);
    for (int i = 0; i < DIMENSION; ++i)
    { size[i] = getImageSize(maskVol, i); }
    size[DIMENSION] = 0;
    for (int i = 0; i < n; ++i) {
      std::vector<int> start(DIMENSION + 1, 0);
      start[DIMENSION] = i;
      masks[i] = extractImage<DIMENSION>(maskVol, start, size);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      if (maskImageFiles.size() > i && maskImageFiles[i] != "NULL")
      { masks[i] = readImage<LabelImage<DIMENSION>>(maskImageFiles[i]); }
    }
  }
  // Find lmaps
  std::vector<std::unordered_map<Label, Label>> lmaps(n);
  parfor(0, n, true, [n, &images, &masks, &lmaps](int i) {
      RegionMap rmap(images[i], masks[i], false);
      rmap.erase(BG_VAL);
      auto& lmap = lmaps[i];
      for (auto const& rp : rmap) {
        std::unordered_map<Label, int> upOverlaps, downOverlaps;
        if (i > 0)
        { stats::getOverlap(downOverlaps, rp.second, images[i - 1]); }
        if (i < n - 1)
        { stats::getOverlap(upOverlaps, rp.second, images[i + 1]); }
        auto uit = upOverlaps.find(rp.first);
        auto dit = downOverlaps.find(rp.first);
        if (uit == upOverlaps.end() && dit == downOverlaps.end())
        { lmap[rp.first] = outputBgVal; }
      }
    }, 0);
  parfor(0, n, true, [&images, &masks, &lmaps](int i) {
      transformImage(images[i], lmaps[i], masks[i], false); }, 0);
  // Write outputs
  if (outputImageFiles.size() == 1) {  // Write to one volume
    auto outputImage = stackImages(images);
    if (relabel) { relabelImage(outputImage, 0); }
    if (write16) {
      castWriteImage<UInt16Image<DIMENSION + 1>>(
          outputImageFiles.front(), outputImage, compress);
    } else {
      writeImage(outputImageFiles.front(), outputImage, compress);
    }
  } else {  // Write to slices
    if (relabel) { relabelImages(images, masks, 0); }
    for (int i = 0; i < n; ++i) {
      if (write16) {
        castWriteImage<UInt16Image<DIMENSION>>(
            outputImageFiles[i], images[i], compress);
      } else {
        writeImage(outputImageFiles[i], images[i], compress);
      }
    }
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("s", bpo::value<std::vector<std::string>>(
          &inputImageFiles)->required(),
       "Input image file name(s) (Use one name for 3D stack)")
      ("m", bpo::value<std::vector<std::string>>(&maskImageFiles),
       "Input mask image file name(s) (Use 'NULL' to bypass) (optional)")
      ("bg", bpo::value<Label>(&outputBgVal),
       "Output background pixel value [default: BG_VAL = 0]")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output label image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("o", bpo::value<std::vector<std::string>>(
          &outputImageFiles)->required(),
       "Output image file name(s) (Use one/multiple file name(s) "
       "to save to one/slice-by-slice file(s)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
