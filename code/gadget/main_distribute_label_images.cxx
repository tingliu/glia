#include "util/image_io.hxx"
#include "util/struct.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> inputImageFiles;
int nOutput;
int areaThreshold;
bool includeBG = false;
bool write16 = false;
bool compress = false;
std::string outputImageFilePattern;

bool operation ()
{
  int nInput = inputImageFiles.size();
  // int nMustKeep = std::min(std::max(0, nInput - 1), nOutput) / 2;
  // std::vector<LabelImage<DIMENSION>::Pointer> inputImages;
  // inputImages.reserve(nInput);
  // for (auto const& f : inputImageFiles)
  // { inputImages.push_back(readImage<LabelImage<DIMENSION>>(f)); }
  // std::vector<std::pair<int, std::vector<int>>> sizes(nInput);
  // for (int i = 0; i < nInput; ++i) {
  //   std::unordered_map<Label, int> cmap;
  //   genCountMap(
  //       cmap, inputImages[i], LabelImage<DIMENSION>::Pointer(nullptr));
  //   if (!includeBG) { cmap.erase(BG_VAL); }
  //   sizes[i].first = i;
  //   sizes[i].second.reserve(cmap.size());
  //   for (auto const& cp : cmap) { sizes[i].second.push_back(cp.second); }
  // }
  // // Sort:
  // //   1. By #regions bigger than areaThreshold
  // //   2. If tied, areaThreshold /= 2
  // //   3. Go to 1
  // std::sort(
  //     sizes.begin(), sizes.end(),
  //     [](std::pair<int, std::vector<int>> const& lhs,
  //        std::pair<int, std::vector<int>> const& rhs) {
  //       int t = areaThreshold;
  //       while (t > 0) {
  //         int nl = std::count_if(
  //             lhs.second.begin(), lhs.second.end(), [t](int x) {
  //               return x > t; });
  //         int nr = std::count_if(
  //             rhs.second.begin(), rhs.second.end(), [t](int x) {
  //               return x > t; });
  //         if (nl < nr) { return true; }
  //         else if (nl > nr) { return false; }
  //         t /= 2;
  //       }
  //       return true;
  //     });
  // // Keep:
  // //   1. If #input == nOutput, keep all
  // //   2. If #input > nOutput, keep first nMustKeep and last nMustKeep,
  // //        sample middle ones
  // //   3. If #input < nOutput, keep first nMustKeep and last nMustKeep,
  // //        duplicate middle ones
  // std::vector<LabelImage<DIMENSION>::Pointer> outputImages(nOutput);
  // if (nInput == nOutput) {
  //   for (int i = 0; i < nOutput; ++i)
  //   { outputImages[i] = inputImages[sizes[i].first]; }
  // } else {
  //   for (int i = 0; i < nMustKeep; ++i) {
  //     outputImages[i] = inputImages[sizes[i].first];
  //     outputImages[nOutput - 1 - i] =
  //         inputImages[sizes[nInput - 1 - i].first];
  //   }
  //   std::vector<int> middleIndices;
  //   for (int i = nMustKeep; i < nInput - nMustKeep; ++i)
  //   { middleIndices.push_back(i); }
  //   int nLeft = nOutput - nMustKeep * 2;
  //   if (nInput > nOutput) {
  //     std::random_shuffle(middleIndices.begin(), middleIndices.end());
  //     middleIndices.resize(nLeft);
  //     std::sort(middleIndices.begin(), middleIndices.end());
  //     for (int i = 0; i < nLeft; ++i) {
  //       outputImages[i + nMustKeep] =
  //           inputImages[sizes[i + nMustKeep].first];
  //     }
  //   } else {
  //     if (middleIndices.empty())
  //     { perr("Unexpected: middle indices empty..."); }
  //     int nDup = (nLeft + 1) / middleIndices.size();
  //     // Prefer images with more regions
  //     auto oit = outputImages.rbegin() + nMustKeep;
  //     auto iit = sizes.rbegin() + nMustKeep;
  //     while (nLeft > 0) {
  //       for (int i = 0; i < nDup; ++i) {
  //         *oit = inputImages[iit->first];
  //         ++oit;
  //         if (--nLeft <= 0) { break; }
  //       }
  //       ++iit;
  //     }
  //   }
  // }
  int nMustKeep = 1;
  std::vector<LabelImage<DIMENSION>::Pointer> inputImages;
  inputImages.reserve(nInput);
  for (auto const& f : inputImageFiles)
  { inputImages.push_back(readImage<LabelImage<DIMENSION>>(f)); }
  std::vector<std::pair<int, std::vector<int>>> sizes(nInput);
  for (int i = 0; i < nInput; ++i) {
    std::unordered_map<Label, int> cmap;
    genCountMap(
        cmap, inputImages[i], LabelImage<DIMENSION>::Pointer(nullptr));
    if (!includeBG) { cmap.erase(BG_VAL); }
    sizes[i].first = i;
    sizes[i].second.reserve(cmap.size());
    for (auto const& cp : cmap) { sizes[i].second.push_back(cp.second); }
  }
  // Sort:
  //   1. By #regions bigger than areaThreshold
  //   2. If tied, areaThreshold /= 2
  //   3. Go to 1
  std::sort(
      sizes.begin(), sizes.end(),
      [](std::pair<int, std::vector<int>> const& lhs,
         std::pair<int, std::vector<int>> const& rhs) {
        int t = areaThreshold;
        while (t > 0) {
          int nl = std::count_if(
              lhs.second.begin(), lhs.second.end(), [t](int x) {
                return x > t; });
          int nr = std::count_if(
              rhs.second.begin(), rhs.second.end(), [t](int x) {
                return x > t; });
          if (nl < nr) { return true; }
          else if (nl > nr) { return false; }
          t /= 2;
        }
        return true;
      });
  // Keep:
  //   1. If #input == nOutput, keep all
  //   2. If #input > nOutput, keep first nMustKeep and last nMustKeep,
  //        sample middle ones
  //   3. If #input < nOutput, keep first nMustKeep and last nMustKeep,
  //        duplicate last one
  std::vector<LabelImage<DIMENSION>::Pointer> outputImages(nOutput);
  if (nInput == nOutput) {
    for (int i = 0; i < nOutput; ++i)
    { outputImages[i] = inputImages[sizes[i].first]; }
  } else if (nInput > nOutput) {
    for (int i = 0; i < nMustKeep; ++i) {
      outputImages[i] = inputImages[sizes[i].first];
      outputImages[nOutput - 1 - i] =
          inputImages[sizes[nInput - 1 - i].first];
    }
    std::vector<int> middleIndices;
    for (int i = nMustKeep; i < nInput - nMustKeep; ++i)
    { middleIndices.push_back(i); }
    int nLeft = nOutput - nMustKeep * 2;
    std::random_shuffle(middleIndices.begin(), middleIndices.end());
    middleIndices.resize(nLeft);
    std::sort(middleIndices.begin(), middleIndices.end());
    for (int i = 0; i < nLeft; ++i) {
      outputImages[i + nMustKeep] =
          inputImages[sizes[i + nMustKeep].first];
    }
  } else {
    for (int i = 0; i < nInput; ++i)
    { outputImages[i] = inputImages[sizes[i].first]; }
    for (int i = nInput; i < nOutput; ++i)
    { outputImages[i] = inputImages[sizes.back().first]; }
  }
  // Write out
  for (int i = 0; i < nOutput; ++i) {
    if (write16) {
      castWriteImage<UInt16Image<DIMENSION>>(
          strprintf(outputImageFilePattern.c_str(), i), outputImages[i],
          compress);
    } else {
      writeImage(
          strprintf(outputImageFilePattern.c_str(), i), outputImages[i],
          compress);
    }
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("i", bpo::value<std::vector<std::string>>(
          &inputImageFiles)->required(), "Input label image file name(s)")
      ("no", bpo::value<int>(&nOutput)->required(),
       "Number of output image files")
      ("t", bpo::value<int>(&areaThreshold)->required(),
       "Area threshold to input images")
      ("bg", bpo::value<bool>(&includeBG),
       "Whether count background pixels in input [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("o", bpo::value<std::string>(&outputImageFilePattern)->required(),
       "Output label image file name pattern");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
