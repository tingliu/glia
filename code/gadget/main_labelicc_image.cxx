#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                std::string const& maskImageFile,
                bool write16, bool compress)
{
  auto image = readImage<LabelImage<DIMENSION>>(inputImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  image = labelIdentityConnectedComponents<LabelImage<DIMENSION>>
      (image, mask, BG_VAL);
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>
        (outputImageFile, image, compress);
  }
  else { writeImage(outputImageFile, image, compress); }
  return true;
}


// Relabel connected components of label image
// Used to relabel 3D label sub-volume cut from bigger volume
int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile, maskImageFile;
  bool write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input label image file name")
      ("mask,m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output label image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, maskImageFile, write16,
                compress)? EXIT_SUCCESS: EXIT_FAILURE;
}
