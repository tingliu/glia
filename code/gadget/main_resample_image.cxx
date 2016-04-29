#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                double factor, std::vector<Int> const& size,
                bool nn, bool compress)
{
  auto info = readImageInfo(inputImageFile);
  switch (info->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        typedef itk::Image<unsigned char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        typedef itk::Image<char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        typedef itk::Image<unsigned short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        typedef itk::Image<short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        typedef itk::Image<unsigned int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        typedef itk::Image<int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        typedef itk::Image<unsigned long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        typedef itk::Image<long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::FLOAT:
      {
        typedef itk::Image<float, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::DOUBLE:
      {
        typedef itk::Image<double, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (size.size() != DIMENSION)
        { image = resampleImage<Image>(image, factor, nn); }
        else {
          itk::Size<DIMENSION> sz;
          for (int i = 0; i < DIMENSION; ++i) { sz[i] = size[i]; }
          image = resampleImage<Image>(image, sz, nn);
        }
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
  double factor = -1.0;
  std::vector<Int> size;
  bool nn, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input real image file name")
      ("factor,f", bpo::value<double>(&factor), "Resample factor")
      ("size,s", bpo::value<std::vector<Int>>(&size)->multitoken(),
       "Resample-to size (overrides factor if set) (e.g. -s 481 321)")
      ("nn,n", bpo::value<bool>(&nn)->required(),
       "Whether to use nearest neighbor (linear otherwise) interpolation")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output real image file name");
  if (!parse(argc, argv, opts)) { return EXIT_FAILURE; }
  if (size.size() != DIMENSION && factor <= 0.0)
  { perr("Error: inappropriate resample size/factor..."); }
  return operation(outputImageFile, inputImageFile, factor, size,
                   nn, compress)? EXIT_SUCCESS: EXIT_FAILURE;
}
