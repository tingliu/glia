#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string inputImageFile;
std::string outputImageFileTemp;
int dim0 = 0;
int dim1 = 1;
bool compress = false;

template <typename T> void
helper ()
{
  typedef itk::Image<T, 3> Image3;
  typedef itk::Image<T, 2> Image2;
  auto image3 = readImage<Image3>(inputImageFile);
  UInt w = getImageSize(image3, 0);
  UInt h = getImageSize(image3, 1);
  UInt d = getImageSize(image3, 2);
  auto image2 = createImage<Image2>({w, h});
  TImageIt<typename Image2::Pointer>
      oit(image2, image2->GetRequestedRegion());
  itk::Index<3> index;
  index.Fill(0);
  for (int ch = 0; ch < d; ++ch) {
    oit.GoToBegin();
    index[2] = ch;
    for (int i = 0; i < h; ++i) {
      index[0] = 0;
      index[1] = i;
      for (int j = 0; j < w; ++j) {
        oit.Set(image3->GetPixel(index));
        ++index[0];
        ++oit;
      }
    }
    writeImage(
        strprintf(outputImageFileTemp.c_str(), ch), image2, compress);
  }
}

// template <typename T> void
// helper ()
// {
//   typedef itk::Image<T, DIMENSION> ImageD;
//   typedef itk::Image<T, 2> Image2;
//   auto imageD = readImage<ImageD>(inputImageFile);
//   UInt w = getImageSize(imageD, dim0);
//   UInt h = getImageSize(imageD, dim1);
//   auto image2 = createImage<Image2>({w, h});
//   TImageCSIIt<typename ImageD::Pointer>
//       iit(imageD, imageD->GetRequestedRegion());
//   iit.SetFirstDirection(dim0);
//   iit.SetSecondDirection(dim1);
//   iit.GoToBegin();
//   TImageIt<typename Image2::Pointer>
//       oit(image2, image2->GetRequestedRegion());
//   int i = 0;
//   while (!iit.IsAtEnd()) {
//     oit.GoToBegin();
//     while (!iit.IsAtEndOfSlice()) {
//       while (!iit.IsAtEndOfLine()) {
//         oit.Set(iit.Get());
//         ++iit;
//         ++oit;
//       }
//       iit.NextLine();
//     }
//     iit.NextSlice();
//     // Debug
//     std::cout << i << std::endl;
//     // ~ Debug
//     writeImage(
//         strprintf(outputImageFileTemp.c_str(), i++), image2, compress);
//   }
// }


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
        helper<unsigned long>();
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
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i",
       bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("d0", bpo::value<int>(&dim0), "First 2D direction [default: 0]")
      ("d1", bpo::value<int>(&dim1), "Second 2D direction [default: 1]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFileTemp)->required(),
       "Output image file name template");
  return parse(argc, argv, opts) && operation();
}
