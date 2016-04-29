#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                double sigma, int kernelWidth, Int dim, bool compress)
{
  switch (readImageInfo(inputImageFile)->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        typedef itk::Image<unsigned char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        typedef itk::Image<char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        typedef itk::Image<unsigned short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        typedef itk::Image<short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        typedef itk::Image<unsigned int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        typedef itk::Image<int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        typedef itk::Image<unsigned long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        typedef itk::Image<long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::FLOAT:
      {
        typedef itk::Image<float, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::DOUBLE:
      {
        typedef itk::Image<double, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        image = dim < 0? blurImage(image, sigma, kernelWidth):
            blurImage(image, sigma, kernelWidth, dim);
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
  double sigma;
  int kernelWidth;
  Int dim = -1;
  bool compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("sigma,s", bpo::value<double>(&sigma)->required(),
       "Sigma of Gaussian kernel")
      ("kernelWidth,k", bpo::value<int>(&kernelWidth)->required(),
       "Width of Gaussian kernel")
      ("dim,d", bpo::value<Int>(&dim),
       "Along dimension (negative to blur in all dimensions) [default: -1]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output label image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, sigma, kernelWidth, dim,
                compress)? EXIT_SUCCESS: EXIT_FAILURE;
}
