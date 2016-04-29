#ifndef _glia_util_image_io_hxx_
#define _glia_util_image_io_hxx_

#include "util/image.hxx"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

namespace glia {

inline itk::ImageIOBase::Pointer
readImageInfo (std::string const& imageFile)
{
  auto imio = itk::ImageIOFactory::CreateImageIO
      (imageFile.c_str(), itk::ImageIOFactory::ReadMode);
  imio->SetFileName(imageFile);
  imio->ReadImageInformation();
  return imio;
  // Usage:
  //   GetNumberOfDimensions()
  //   GetComponentTypeAsString(GetComponentType())
  //   GetImageSizeInPixels()...
}

inline std::vector<Int>
getImageSize (itk::ImageIOBase::Pointer const& info)
{
  auto D = info->GetNumberOfDimensions();
  std::vector<Int> ret(D);
  for (int i = 0; i < D; ++i) { ret[i] = info->GetDimensions(i); }
  return ret;
}

template <typename TImage> typename TImage::Pointer
readImage (std::string const& imageFile)
{
  typedef itk::ImageFileReader<TImage> Reader;
  auto reader = Reader::New();
  reader->SetFileName(imageFile);
  reader->Update();
  return reader->GetOutput();
}


template <typename TImagePtr> void
writeImage (std::string const& imageFile, TImagePtr const& image,
            bool compress)
{
  typedef itk::ImageFileWriter<TImage<TImagePtr>> Writer;
  auto writer = Writer::New();
  writer->SetFileName(imageFile);
  writer->SetInput(image);
  writer->SetUseCompression(compress);
  writer->Update();
}


template <typename TImageOut, typename TImagePtrIn> void
castWriteImage (std::string const& imageFile, TImagePtrIn const& image,
                bool compress)
{ writeImage(imageFile, castImage<TImageOut>(image), compress); }


template <typename TVecImage> typename TVecImage::Pointer
readVectorImage (std::string const& imageFile)
{
  typedef typename TVecImage::PixelType::ValueType Value;
  const UInt D = TVecImage::ImageDimension + 1;
  typedef itk::Image<Value, D> Image;
  auto image = readImage<Image>(imageFile);
  UInt depth = getImageSize(image, D - 1);
  itk::Index<D - 1> _index;
  _index.Fill(0);
  itk::Size<D - 1> _size;
  for (int i = 0; i < D - 1; ++i) { _size[i] = getImageSize(image, i); }
  auto ret = createVectorImage<TVecImage>
      (itk::ImageRegion<D - 1>(_index, _size), depth);
  itk::VariableLengthVector<Value> vec;
  vec.SetSize(depth);
  itk::ImageRegionConstIteratorWithIndex<Image>
      iit(image, image->GetRequestedRegion());
  for (itk::ImageRegionIterator<TVecImage>
           vit(ret, ret->GetRequestedRegion());
       !vit.IsAtEnd(); ++vit, ++iit) {
    auto iidx0 = iit.GetIndex();
    auto iidx1 = iidx0;
    iidx1[D - 1] +=depth - 1;
    itk::LineConstIterator<Image> lit(image, iidx0, iidx1);
    for (int i = 0; i < depth; ++lit, ++i) { vec[i] = lit.Get(); }
    vit.Set(vec);
  }
  return ret;
}


template <typename TVecImagePtr> void
writeVectorImage (std::string const& imageFile,
                  TVecImagePtr const& vecImage)
{
  const UInt D = TImage<TVecImagePtr>::ImageDimension + 1;
  UInt depth = vecImage->GetVectorLength();
  typedef TVecImageVal<TVecImagePtr> Value;
  typedef itk::Image<Value, D> Image;
  itk::Index<D> _index;
  _index.Fill(0);
  itk::Size<D> _size;
  for (int i = 0; i < D - 1; ++i)
  { _size[i] = getImageSize(vecImage, i); }
  _size[D - 1] = depth;
  auto image = createImage<Image>(itk::ImageRegion<D>(_index, _size));
  itk::ImageRegionIteratorWithIndex<Image>
      iit(image, image->GetRequestedRegion());
  for (itk::ImageRegionConstIterator<TImage<TVecImagePtr>>
           vit(vecImage, vecImage->GetRequestedRegion());
       !vit.IsAtEnd(); ++vit, ++iit) {
    auto const& vec = vit.Get();
    auto iidx0 = iit.GetIndex();
    auto iidx1 = iidx0;
    iidx1[D - 1] += depth - 1;
    itk::LineIterator<Image> lit(image, iidx0, iidx1);
    for (int i = 0; i < depth; ++i, ++lit) { lit.Set(vec[i]); }
  }
  writeImage(imageFile, image);
}

};

#endif
