#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& boundaryImageFile,
                std::vector<std::string> const& segImageFiles,
                std::vector<std::string> const& bcImageFiles,
                bool doubleSize, bool invert, bool compress)
{
  RealImage<DIMENSION>::Pointer outputImage, bImage;
  int n;
  if (bcImageFiles.empty()) { // Accumulate binary boundaries
    auto f = [](TImageCLIIt<LabelImage<DIMENSION>::Pointer> const& iit0,
                TImageCLIIt<LabelImage<DIMENSION>::Pointer> const& iit1)
        -> Real { return 1.0; };
    n = segImageFiles.size();
    for (int i = 0; i < n; ++i) {
      auto segImage = readImage<LabelImage<DIMENSION>>(segImageFiles[i]);
      if (i == 0) {
        bImage =
            genBoundaryImage<RealImage<DIMENSION>>(segImage, f);
        outputImage = cloneImage(bImage);
      }
      else {
        genBoundaryImage(bImage, segImage, f);
        addImage(outputImage, bImage);
      }
    }
  }
  else if (segImageFiles.size() == 1) { // Accumulate boundary confidences
    auto segImage = readImage<LabelImage<DIMENSION>>(segImageFiles[0]);
    RealImage<DIMENSION>::Pointer bcImage;
    auto f = [&bcImage]
        (TImageCLIIt<LabelImage<DIMENSION>::Pointer> const& iit0,
         TImageCLIIt<LabelImage<DIMENSION>::Pointer> const& iit1) -> Real
        {
          return std::max(bcImage->GetPixel(iit0.GetIndex()),
                          bcImage->GetPixel(iit1.GetIndex()));
        };
    n = bcImageFiles.size();
    for (int i = 0; i < n; ++i) {
      bcImage = readImage<RealImage<DIMENSION>>(bcImageFiles[i]);
      if (i == 0) {
        bImage =
            genBoundaryImage<RealImage<DIMENSION>>(segImage, f);
        outputImage = cloneImage(bImage);
      }
      else {
        genBoundaryImage(bImage, segImage, f);
        addImage(outputImage, bImage);
      }
    }
  }
  else { perr("Error: neither ignore bcImage nor use one segi..."); }
  if (!doubleSize)
  { outputImage = sampleImage(outputImage, {2, 2}, {2, 2}); }
  for (TImageIt<RealImage<DIMENSION>::Pointer>
           iit(outputImage, outputImage->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) { iit.Value() /= (double)n; }
  if (invert) {
    for (TImageIt<RealImage<DIMENSION>::Pointer>
             iit(outputImage, outputImage->GetRequestedRegion());
         !iit.IsAtEnd(); ++iit) { iit.Value() = 1.0 - iit.Value(); }
  }
  writeImage(boundaryImageFile, outputImage, compress);
  return true;
}


int main (int argc, char* argv[])
{
  std::string boundaryImageFile;
  std::vector<std::string> segImageFiles, bcImageFiles;
  bool doubleSize = true, invert = false, compress = false;
  bpo::options_description
      opts("Usage: [1] use all segs, or [2] use segi and bcImages");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s",
       bpo::value<std::vector<std::string>>(&segImageFiles)->required(),
       "Input (initial) segmentation image file name(s)")
      ("bcImage,c", bpo::value<std::vector<std::string>>(&bcImageFiles),
       "Input boundary confidence image file names (optional)")
      ("doubleSize,d", bpo::value<bool>(&doubleSize),
       "Whether to double image size [default: true]")
      ("invert,i", bpo::value<bool>(&invert),
       "Whether to invert boundary image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("bImage,b", bpo::value<std::string>(&boundaryImageFile)->required(),
       "Output double-sized averaged boundary image file name");
  return
      parse(argc, argv, opts) &&
      operation(boundaryImageFile, segImageFiles,
                bcImageFiles, doubleSize, invert, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
