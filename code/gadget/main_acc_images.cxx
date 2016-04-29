#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::vector<std::string> const& inputImageFiles,
                bool avg, bool compress)
{
  int n = inputImageFiles.size();
  auto resImage = readImage<RealImage<DIMENSION>>(inputImageFiles[0]);
  for (int i = 1; i < n; ++i) {
    addImage(resImage, readImage<RealImage<DIMENSION>>(inputImageFiles[i]));
  }
  if (avg) { multiplyImage(resImage, 1.0 / n); }
  writeImage(outputImageFile, resImage, compress);
  return true;
}


int main (int argc, char* argv[])
{
  std::string outputImageFile;
  std::vector<std::string> inputImageFiles;
  bool avg = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i",
       bpo::value<std::vector<std::string>>(&inputImageFiles)->required(),
       "Input image file names")
      ("avg,a", bpo::value<bool>(&avg),
       "Whether to average images [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFiles, avg, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
