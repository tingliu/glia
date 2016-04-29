#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

bool operation (std::string const& outputPatchFile,
                std::string const& valImageFile,
                std::string const& maskImageFile,
                std::vector<int> const& patchRadius)
{
  std::vector<std::vector<Real>> patches;
  auto valImage = readImage<RealImage<DIMENSION>>(valImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  getImagePatches(patches, valImage, mask, patchRadius);
  writeData(outputPatchFile, patches, " ", "\n", FLT_PREC);
  return true;
}


int main (int argc, char* argv[])
{
  std::string outputPatchFile, valImageFile, maskImageFile;
  std::vector<int> patchRadius;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("valImage,i", bpo::value<std::string>(&valImageFile)->required(),
       "Input value image file name")
      ("mask,m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name")
      ("radius,r", bpo::value<std::vector<int>>(&patchRadius)->multitoken(),
       "Patch radius (e.g. -r 7 7)")
      ("patch,o", bpo::value<std::string>(&outputPatchFile)->required(),
       "Output patch file name");
  return
      parse(argc, argv, opts) &&
      operation(outputPatchFile, valImageFile, maskImageFile, patchRadius)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
