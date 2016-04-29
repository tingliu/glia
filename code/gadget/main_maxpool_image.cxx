#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                std::vector<Int> const& skipDims, int n, bool compress)
{
  std::unordered_set<Int> sd;
  for (auto d: skipDims) { sd.insert(d); }
  switch (readImageInfo(inputImageFile)->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        typedef itk::Image<unsigned char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        typedef itk::Image<char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        typedef itk::Image<unsigned short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        typedef itk::Image<short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        typedef itk::Image<unsigned int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        typedef itk::Image<int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        typedef itk::Image<unsigned long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        typedef itk::Image<long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::FLOAT:
      {
        typedef itk::Image<float, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::DOUBLE:
      {
        typedef itk::Image<double, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        for (int i = 0; i < n; ++i)
        { image = maxPoolImage(image, sd); }
        writeImage(outputImageFile, image, compress);
        break;
      }
    default: perr("Error: unsupported image pixel type...");
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile;
  std::vector<Int> skipDims;
  int n = 1;
  bool compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input real image file name")
      ("skipDims,d", bpo::value<std::vector<Int>>(&skipDims),
       "Skipped dimensions (e.g. use -d 2 to skip z) (optional)")
      ("nPool,n", bpo::value<int>(&n), "Number of pooling [default: 1]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output real image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, skipDims, n, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
