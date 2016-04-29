#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                double factor, std::vector<int> const& size)
{
  auto image = readImage<RgbImage>(inputImageFile);
  if (size.size() != DIMENSION)
  { image = resampleVectorImage<RgbImage>(image, factor, false); }
  else {
    itk::Size<DIMENSION> sz;
    for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
    image = resampleVectorImage<RgbImage>(image, sz, false);
  }
  writeImage(outputImageFile, image, false);
  return true;
}


int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile;
  double factor = -1.0;
  std::vector<int> size;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input RGB image file name")
      ("factor,f", bpo::value<double>(&factor), "Resample factor")
      ("size,s", bpo::value<std::vector<int>>(&size)->multitoken(),
       "Resample-to size (overrides factor if set)")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output RGB image file name");
  if (!parse(argc, argv, opts)) { return EXIT_FAILURE; }
  if (size.size() != DIMENSION && factor <= 0.0)
  { perr("Error: inappropriate resample size/factor..."); }
  return operation(outputImageFile, inputImageFile, factor, size)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
