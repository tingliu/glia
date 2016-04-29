#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string inputImageFile;
double lower = 0.0;
double upper;
Label inside = 1;
Label outside = 0;
bool doLabelCC = false;
bool doSubSample = false;
bool doRelabel = false;
bool write16 = false;
bool compress = false;
std::string outputImageFile;

bool operation ()
{
  auto inputImage = readImage<RealImage<DIMENSION>>(inputImageFile);
  auto outputImage = thresholdImage<LabelImage<DIMENSION>>(
      inputImage, lower, upper, inside, outside);
  if (doLabelCC) {
    outputImage =
        labelConnectedComponents<LabelImage<DIMENSION>>(outputImage);
  }
  if (doSubSample) {
    outputImage = sampleImage(outputImage, {2, 2}, {2, 2});
    dilateImage(outputImage, LabelImage<DIMENSION>::Pointer());
  }
  if (doRelabel) { relabelImage(outputImage, 0); }
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>(
        outputImageFile, outputImage, compress);
  } else { writeImage(outputImageFile, outputImage, compress); }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("lower,l", bpo::value<double>(&lower),
       "Lower bound threshold [default: 0.0]")
      ("upper,t", bpo::value<double>(&upper), "Upper bound threshold")
      ("iv", bpo::value<Label>(&inside), "Inside value [default: 1]")
      ("ov", bpo::value<Label>(&outside), "Outside value [default: 0]")
      ("labelCC,c", bpo::value<bool>(&doLabelCC),
       "Whether to label connected components [default: false]")
      ("subs,s", bpo::value<bool>(&doSubSample),
       "Whether to subsample image (to half size in BSDS style) "
       "[default: false]")
      ("relabel,r", bpo::value<bool>(&doRelabel),
       "Whether to relabel image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
