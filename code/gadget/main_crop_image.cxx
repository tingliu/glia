#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string inputImageFile;
std::vector<int> startIndex;
std::vector<UInt> size;
bool compress = false;
std::string outputImageFile;

template <typename T> void
helper ()
{
  typedef itk::Image<T, DIMENSION> Image;
  auto srcRegion = createItkImageRegion<DIMENSION>(startIndex, size);
  auto dstRegion = createItkImageRegion<DIMENSION>(size);
  auto inputImage = readImage<Image>(inputImageFile);
  auto outputImage = createImage<Image>(dstRegion);
  copyImage(outputImage, inputImage, srcRegion, dstRegion.GetIndex());
  writeImage(outputImageFile, outputImage, compress);
}

bool operation ()
{

  switch (readImageInfo(inputImageFile)->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        helper<unsigned char>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        helper<char>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        helper<unsigned short>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        helper<short>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        helper<unsigned int>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        helper<int>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        helper<unsigned long>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        helper<long>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::FLOAT:
      {
        helper<float>();
        break;
      }
    case itk::ImageIOBase::IOComponentType::DOUBLE:
      {
        helper<double>();
        break;
      }
    default: perr("Error: unsupported image pixel type...");
  }
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i",
       bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("start,x", bpo::value<std::vector<int>>(&startIndex)->multitoken(),
       "Starting index of desired region")
      ("size,s", bpo::value<std::vector<UInt>>(&size)->multitoken(),
       "Size of desired region")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return parse(argc, argv, opts) && operation();
}
