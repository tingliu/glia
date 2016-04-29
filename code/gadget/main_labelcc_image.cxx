#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                bool invert, bool relabel, bool write16, bool compress)
{
  auto image = readImage<LabelImage<DIMENSION>>(inputImageFile);
  if (invert) {
    TImageIt<LabelImage<DIMENSION>::Pointer>
        iit(image, image->GetRequestedRegion());
    while (!iit.IsAtEnd()) {
      if (iit.Value() == 0) { iit.Value() = 1; }
      else { iit.Value() = 0; }
      ++iit;
    }
  }
  image = labelConnectedComponents<LabelImage<DIMENSION>>(image);
  if (relabel) { relabelImage(image, 0); }
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>
        (outputImageFile, image, compress);
  }
  else { writeImage(outputImageFile, image, compress); }
  return true;
}


int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile;
  bool invert = false, relabel = false, write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input label image file name")
      ("invert,v", bpo::value<bool>(&invert),
       "Whether to invert input image [default: false]")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output label image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, invert, relabel, write16,
                compress)? EXIT_SUCCESS: EXIT_FAILURE;
}
