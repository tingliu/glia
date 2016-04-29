#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                unsigned char diffThreshold, bool compress)
{
  switch (readImageInfo(inputImageFile)->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        typedef itk::Image<unsigned char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        typedef itk::Image<char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        typedef itk::Image<unsigned short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        typedef itk::Image<short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        typedef itk::Image<unsigned int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        typedef itk::Image<int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        typedef itk::Image<unsigned long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        typedef itk::Image<long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        labelScalarConnectedComponents<Image>
            (image, (Image::PixelType)diffThreshold);
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
  unsigned char diffThreshold = 0;
  bool compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("diff,d", bpo::value<unsigned char>(&diffThreshold),
       "Intensity difference threshold [default: 0]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, diffThreshold, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
