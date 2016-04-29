#ifndef _glia_util_image_hxx_
#define _glia_util_image_hxx_

#include "glia_image.hxx"
#include "type/neighbor.hxx"
#include "util/struct.hxx"
#include "util/container.hxx"
#include "itkCastImageFilter.h"
#include "itkVectorImageToImageAdaptor.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkScalarConnectedComponentImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkVectorResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkBinaryThinningImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkSliceBySliceImageFilter.h"
#include "itkLabelImageToLabelMapFilter.h"
#include "itkLabelMapOverlayImageFilter.h"

namespace glia {

template <typename TImagePtr> UInt
getImagePixelNumber (TImagePtr const& image)
{
  auto size = image->GetRequestedRegion().GetSize();
  UInt ret = 1;
  for (int i = 0; i < size.GetSizeDimension(); ++i) { ret *= size[i]; }
  return ret;
}


template <typename TSize0, typename TSize1> bool
compareSize (TSize0 const& sz0, TSize1 const& sz1, int D)
{
  bool ret = true;
  int cnt = 0;
  for (int i = 0; i < D; ++i) {
    if (sz1[i] > sz0[i]) { return false; }
    else if (sz1[i] == sz0[i]) { ++cnt; }
  }
  if (cnt == D) { return false; }
  return ret;
}


template <UInt D> inline itk::ImageRegion<D>
createItkImageRegion (std::initializer_list<UInt> const& size)
{
  assert("Error: incorrect size dimension..." && size.size() == D);
  itk::Index<D> _index;
  _index.Fill(0);
  itk::Size<D> _size;
  std::copy(size.begin(), size.begin() + D, _size.m_Size);
  return itk::ImageRegion<D>(_index, _size);
}


template <UInt D> inline itk::ImageRegion<D>
createItkImageRegion (std::vector<UInt> const& size)
{
  assert("Error: incorrect size dimension..." && size.size() == D);
  itk::Index<D> _index;
  _index.Fill(0);
  itk::Size<D> _size;
  std::copy(size.begin(), size.begin() + D, _size.m_Size);
  return itk::ImageRegion<D>(_index, _size);
}



template <UInt D> inline itk::ImageRegion<D>
createItkImageRegion (
    std::vector<int> const& startIndex, std::vector<UInt> const& size)
{
  assert("Error: incorrect index dimension..." && startIndex.size() == D);
  assert("Error: incorrect size dimension..." && size.size() == D);
  itk::Index<D> _index;
  for (int i = 0; i < D; ++i) { _index[i] = startIndex[i]; }
  itk::Size<D> _size;
  std::copy(size.begin(), size.begin() + D, _size.m_Size);
  return itk::ImageRegion<D>(_index, _size);
}



template <typename TImage> typename TImage::Pointer
createImage (itk::ImageRegion<TImage::ImageDimension> const& region)
{
  auto ret = TImage::New();
  ret->SetRegions(region);
  ret->Allocate();
  return ret;
}


template <typename TImage> typename TImage::Pointer
createImage (itk::ImageRegion<TImage::ImageDimension> const& region,
             typename TImage::PixelType val)
{
  auto ret = createImage<TImage>(region);
  ret->FillBuffer(val);
  return ret;
}


template <typename TImage> typename TImage::Pointer
createImage (std::initializer_list<UInt> const& size)
{
  return createImage<TImage>
      (createItkImageRegion<TImage::ImageDimension>(size));
}


template <typename TImage> typename TImage::Pointer
createImage (std::initializer_list<UInt> const& size,
             typename TImage::PixelType val)
{
  auto ret = createImage<TImage>(size);
  ret->FillBuffer(val);
  return ret;
}


template <typename TVecImage> typename TVecImage::Pointer
createVectorImage
(itk::ImageRegion<TVecImage::ImageDimension> const& region, UInt depth)
{
  auto ret = TVecImage::New();
  ret->SetRegions(region);
  ret->SetVectorLength(depth);
  ret->Allocate();
  return ret;
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer castImage (TImagePtrIn const& image)
{
  typedef itk::CastImageFilter<TImage<TImagePtrIn>, TImageOut> Caster;
  auto caster = Caster::New();
  caster->SetInput(image);
  caster->Update();
  return caster->GetOutput();
}


template <typename TImagePtr> inline UInt
getImageSize (TImagePtr const& image)
{
  UInt ret = 1;
  for (int i = 0; i < TImage<TImagePtr>::ImageDimension; ++i)
  { ret *= image->GetRequestedRegion().GetSize()[i]; }
  return ret;
}


template <typename TImagePtr> inline UInt
getImageSize (TImagePtr const& image, int dimension)
{ return image->GetRequestedRegion().GetSize()[dimension]; }


template <typename TImagePtr> UInt
getImageVolume (TImagePtr const& image)
{
  UInt ret = 1.0;
  for (int i = 0; i < TImage<TImagePtr>::ImageDimension; ++i)
  { ret *= getImageSize(image, i); }
  return ret;
}


template <typename TImagePtr> double
getImageDiagonal (TImagePtr const& image)
{
  double ret = 0.0;
  for (int i = 0; i < TImage<TImagePtr>::ImageDimension; ++i) {
    double l = getImageSize(image, i);
    ret += l * l;
  }
  return std::sqrt(ret);
}


template <typename TVecImagePtr> inline
typename itk::VectorImageToImageAdaptor
<TVecImageVal<TVecImagePtr>, TImage<TVecImagePtr>::ImageDimension>
::Pointer
getImageComponent (TVecImagePtr const& image, UInt component)
{
  auto ret = itk::VectorImageToImageAdaptor
      <TVecImageVal<TVecImagePtr>, TImage<TVecImagePtr>::ImageDimension>
      ::New();
  ret->SetExtractComponentIndex(component);
  ret->SetImage(image);
  return ret;
}


// Every orignal value has to have correspondence
template <typename TImagePtr, typename TMaskPtr> void
transformImage (
    TImagePtr& image, std::unordered_map<TImageVal<TImagePtr>,
    TImageVal<TImagePtr>> const& lmap, TMaskPtr const& mask)
{
  for (TImageIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (mask.IsNull() ||
        mask->GetPixel(iit.GetIndex()) != MASK_OUT_VAL)
    { iit.Set(lmap.find(iit.Get())->second); }
  }
}


// If fillMissing == true, use fill pixels that miss label mappings
// with BG_VAL
template <typename TImagePtr, typename TMaskPtr> void
transformImage (
    TImagePtr& image, std::unordered_map<TImageVal<TImagePtr>,
    TImageVal<TImagePtr>> const& lmap, TMaskPtr const& mask,
    bool fillMissing)
{
  for (TImageIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (mask.IsNull() ||
        mask->GetPixel(iit.GetIndex()) != MASK_OUT_VAL) {
      auto lit = lmap.find(iit.Get());
      if (lit != lmap.end()) { iit.Set(lit->second); }
      else if (fillMissing) { iit.Set(BG_VAL); }
    }
  }
}


// Every orignal value must have correspondence
template <typename TImagePtr, typename TRegionMap> void
transformImage (TImagePtr& image, TRegionMap const& rmap,
                std::unordered_map<TImageVal<TImagePtr>,
                TImageVal<TImagePtr>> const& lmap)
{
  for (auto const& lp: lmap) {
    auto rit = rmap.find(lp.first);
    if (rit != rmap.end()) {
      rit->second.traverse
          ([&image, &lp](typename TRegionMap::Region::Point const& p)
           { image->SetPixel(p, lp.second); });
    }
  }
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
thresholdImage (TImagePtrIn const& image,
                TImageVal<TImagePtrIn> const& lowerThreshold,
                TImageVal<TImagePtrIn> const& upperThreshold,
                typename TImageOut::PixelType const& insideVal,
                typename TImageOut::PixelType const& outsideVal)
{
  auto filter = itk::BinaryThresholdImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  filter->SetInput(image);
  filter->SetLowerThreshold(lowerThreshold);
  filter->SetUpperThreshold(upperThreshold);
  filter->SetInsideValue(insideVal);
  filter->SetOutsideValue(outsideVal);
  filter->Update();
  return filter->GetOutput();
}


template <typename TImagePtr> void
thresholdImageInPlace (TImagePtr& image,
                       TImageVal<TImagePtr> const& lowerThreshold,
                       TImageVal<TImagePtr> const& upperThreshold,
                       TImageVal<TImagePtr> const& insideVal,
                       TImageVal<TImagePtr> const& outsideVal)
{
  auto thresholder = itk::BinaryThresholdImageFilter
      <TImage<TImagePtr>, TImage<TImagePtr>>::New();
  thresholder->SetInput(image);
  thresholder->InPlaceOn();
  thresholder->SetLowerThreshold(lowerThreshold);
  thresholder->SetUpperThreshold(upperThreshold);
  thresholder->SetInsideValue(insideVal);
  thresholder->SetOutsideValue(outsideVal);
  thresholder->Update();
  image = thresholder->GetOutput();
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
labelConnectedComponents (TImagePtrIn const& image)
{
  auto labeler = itk::ConnectedComponentImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  labeler->SetInput(image);
  labeler->SetBackgroundValue(BG_VAL);
  labeler->SetFullyConnected(false);
  labeler->Update();
  return labeler->GetOutput();
}


// Does not work for label images?
template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
labelScalarConnectedComponents
(TImagePtrIn const& image, TImageVal<TImagePtrIn> const& diffThreshold)
{
  auto filter = itk::ScalarConnectedComponentImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  filter->SetDistanceThreshold(diffThreshold);
  filter->SetInput(image);
  filter->Update();
  return filter->GetOutput();
}


// Relabel connected components of label image
// Used to relabel 3D label sub-volume cut from bigger volume
template <typename TImageOut, typename TImagePtrIn, typename TMaskPtr>
typename TImageOut::Pointer
labelIdentityConnectedComponents
(TImagePtrIn const& image, TMaskPtr const& mask,
 TImageVal<TImagePtrIn> bgVal)
{
  const UInt D = TImageOut::ImageDimension;
  auto ret = createImage<TImageOut>(image->GetRequestedRegion(), bgVal);
  TImageCIIt<TImagePtrIn> iit(image, image->GetRequestedRegion());
  TImageIt<typename TImageOut::Pointer>
      oit(ret, ret->GetRequestedRegion());
  auto valToAssign = bgVal + 1;
  std::queue<itk::Index<D>> iq;
  while (!iit.IsAtEnd()) {
    auto val = iit.Value();
    auto idx = iit.GetIndex();
    if ((mask.IsNull() || mask->GetPixel(idx) != MASK_OUT_VAL) &&
        val != bgVal && oit.Get() == bgVal) {
      iq.push(idx);
      while (!iq.empty()) {
        auto const& i = iq.front();
        ret->SetPixel(i, valToAssign);
        traverseNeighbors(i, image->GetRequestedRegion(), mask,
                          [&iq, &image, &ret, val, valToAssign, bgVal]
                          (itk::Index<D> const& p)
                          {
                            if (image->GetPixel(p) == val
                                && ret->GetPixel(p) == bgVal) {
                              iq.push(p);
                              ret->SetPixel(p, valToAssign);
                            }
                          });
        iq.pop();
      }
      ++valToAssign;
    }
    ++iit;
    ++oit;
  }
  return ret;
}


template <typename TImagePtr> void
relabelImage (TImagePtr& image, int minSize)
{
  typedef TImage<TImagePtr> Image;
  auto relabeler = itk::RelabelComponentImageFilter<Image, Image>::New();
  relabeler->InPlaceOn();
  relabeler->SetInput(image);
  if (minSize > 0) { relabeler->SetMinimumObjectSize(minSize); }
  relabeler->Update();
  image = relabeler->GetOutput();
}


template <typename TImagePtr> TImagePtr
blurImage (TImagePtr const& image, double sigma, int kernelWidth)
{
  typedef TImage<TImagePtr> Image;
  auto g = itk::DiscreteGaussianImageFilter<Image, Image>::New();
  g->SetInput(image);
  g->SetVariance(sigma * sigma);
  g->SetMaximumKernelWidth(kernelWidth);
  g->Update();
  return g->GetOutput();
}


// Blur image slice by slice along specified dimension
template <typename TImagePtr> TImagePtr
blurImage (TImagePtr const& image, double sigma, int kernelWidth,
           int dim)
{
  typedef TImage<TImagePtr> ImageHD;
  typedef itk::Image<TImageVal<TImagePtr>, ImageHD::ImageDimension - 1>
      ImageLD;
  auto blurrer =
      itk::DiscreteGaussianImageFilter<ImageLD, ImageLD>::New();
  blurrer->SetVariance(sigma * sigma);
  blurrer->SetMaximumKernelWidth(kernelWidth);
  auto slicer = itk::SliceBySliceImageFilter<ImageHD, ImageHD>::New();
  slicer->SetFilter(blurrer);
  slicer->SetDimension(dim);
  slicer->SetInput(image);
  slicer->Update();
  return slicer->GetOutput();
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
shrinkImage (TImagePtrIn const& image, std::vector<int> const& factors)
{
  auto filter =
      itk::ShrinkImageFilter<TImage<TImagePtrIn>, TImageOut>::New();
  filter->SetInput(image);
  for (int i = 0; i < TImage<TImagePtrIn>::ImageDimension; ++i)
  { filter->SetShrinkFactor(i, factors[i]); }
  filter->Update();
  return filter->GetOutput();
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
resampleImage
(TImagePtrIn image, typename TImageOut::SizeType const& size,
 typename TImageOut::SpacingType const& spacing,
 bool nearestNeighborInterpolation)
{
  const uint D = TImage<TImagePtrIn>::ImageDimension;
  auto resampler =
      itk::ResampleImageFilter<TImage<TImagePtrIn>, TImageOut>::New();
  resampler->SetInput(image);
  resampler->SetSize(size);
  resampler->SetOutputSpacing(spacing);
  resampler->SetTransform(itk::IdentityTransform<double, D>::New());
  if (nearestNeighborInterpolation) {
    resampler->SetInterpolator
        (itk::NearestNeighborInterpolateImageFunction
         <TImage<TImagePtrIn>, double>::New());
  }
  resampler->Update();
  return resampler->GetOutput();
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
resampleImage (TImagePtrIn const& image, double factor,
               bool nearestNeighborInterpolation)
{
  const uint D = TImage<TImagePtrIn>::ImageDimension;
  auto size = image->GetRequestedRegion().GetSize();
  auto spacing = image->GetSpacing();
  for (int i = 0; i < D; ++i) {
    size[i] = std::ceil(size[i] * factor);
    spacing[i] /= factor;
  }
  return resampleImage<TImageOut>
      (image, size, spacing, nearestNeighborInterpolation);
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
resampleImage (TImagePtrIn const& image,
               typename TImageOut::SizeType const& size,
               bool nearestNeighborInterpolation)
{
  const uint D = TImage<TImagePtrIn>::ImageDimension;
  auto sizeIn = image->GetRequestedRegion().GetSize();
  auto spacing = image->GetSpacing();
  for (int i = 0; i < D; ++i)
  { spacing[i] /= (double)size[i] / sizeIn[i]; }
  return resampleImage<TImageOut>
      (image, size, spacing, nearestNeighborInterpolation);
}


template <typename TVecImageOut, typename TVecImagePtrIn>
typename TVecImageOut::Pointer
resampleVectorImage (TVecImagePtrIn const& image,
                     typename TVecImageOut::SizeType const& size,
                     typename TVecImageOut::SpacingType const& spacing,
                     bool nearestNeighborInterpolation)
{
  const uint D = TImage<TVecImagePtrIn>::ImageDimension;
  auto resampler = itk::VectorResampleImageFilter
      <TImage<TVecImagePtrIn>, TVecImageOut>::New();
  resampler->SetInput(image);
  resampler->SetSize(size);
  resampler->SetOutputSpacing(spacing);
  resampler->SetTransform(itk::IdentityTransform<double, D>::New());
  if (nearestNeighborInterpolation) {
    resampler->SetInterpolator
        (itk::VectorNearestNeighborInterpolateImageFunction
         <TImage<TVecImagePtrIn>, double>::New());
  }
  resampler->Update();
  return resampler->GetOutput();
}


template <typename TVecImageOut, typename TVecImagePtrIn>
typename TVecImageOut::Pointer
resampleVectorImage (TVecImagePtrIn const& image, double factor,
                     bool nearestNeighborInterpolation)
{
  const uint D = TImage<TVecImagePtrIn>::ImageDimension;
  auto size = image->GetRequestedRegion().GetSize();
  auto spacing = image->GetSpacing();
  for (int i = 0; i < D; ++i) {
    size[i] = std::ceil(size[i] * factor);
    spacing[i] /= factor;
  }
  return resampleVectorImage<TVecImageOut>
      (image, size, spacing, nearestNeighborInterpolation);
}


template <typename TVecImageOut, typename TVecImagePtrIn>
typename TVecImageOut::Pointer
resampleVectorImage (TVecImagePtrIn const& image,
                     typename TVecImageOut::SizeType const& size,
                     bool nearestNeighborInterpolation)
{
  const uint D = TImage<TVecImagePtrIn>::ImageDimension;
  auto sizeIn = image->GetRequestedRegion().GetSize();
  auto spacing = image->GetSpacing();
  for (int i = 0; i < D; ++i)
  { spacing[i] /= (double)size[i] / sizeIn[i]; }
  return resampleVectorImage<TVecImageOut>
      (image, size, spacing, nearestNeighborInterpolation);
}


template <typename TImagePtr> void
rescaleImage (TImagePtr& image, TImageVal<TImagePtr> const& outputMin,
              TImageVal<TImagePtr> const& outputMax)
{
  auto filter =
      itk::RescaleIntensityImageFilter<TImage<TImagePtr>>::New();
  filter->SetInput(image);
  filter->SetOutputMinimum(outputMin);
  filter->SetOutputMaximum(outputMax);
  filter->Update();
  image = filter->GetOutput();
}


template <typename TImagePtr> TImagePtr
maxPoolImage
(TImagePtr const& image, std::unordered_set<Int> const& skipDims)
{
  const UInt D = TImage<TImagePtr>::ImageDimension;
  auto reg0 = image->GetRequestedRegion();
  itk::ImageRegion<D> reg1;
  reg1.GetModifiableIndex().Fill(0);
  for (int i = 0; i < D; ++i) {
    reg1.GetModifiableSize()[i] = skipDims.count(i) > 0?
        reg0.GetSize()[i]: std::ceil(reg0.GetSize()[i] / 2.0);
  }
  TImagePtr image1 = createImage<TImage<TImagePtr>>(reg1);
  auto idx0 = reg0.GetIndex();
  std::array<Int, D> delta, sizes;
  delta.fill(0);
  for (int i = 0; i < D; ++i) { sizes[i] = getImageSize(image, i); }
  for (TImageIt<TImagePtr> iit1(image1, reg1); !iit1.IsAtEnd(); ++iit1) {
    reg0.SetIndex(idx0);
    int carry = 1;
    for (int i = 0; i < D; ++i) {
      int step;
      if (skipDims.count(i) > 0) {
        step = 1;
        delta[i] += carry;
        carry = delta[i] / sizes[i];
      }
      else {
        delta[i] += (carry << 1);
        carry = delta[i] / sizes[i];
        if (idx0[i] < sizes[i] - 1) { step = 2; }
        else {
          step = 1;
          --delta[i];
        }
      }
      delta[i] %= sizes[i];
      reg0.GetModifiableSize()[i] = step;
      idx0[i] = image->GetRequestedRegion().GetIndex()[i] + delta[i];
    }
    TImageCIt<TImagePtr> iit0(image, reg0);
    TImageVal<TImagePtr> maxi = iit0.Get();
    while (!(++iit0).IsAtEnd()) { maxi = std::max(maxi, iit0.Get()); }
    iit1.Set(maxi);
  }
  return image1;
}


// In-place: image0 += image1
template <typename TImagePtr> void
addImage (TImagePtr& image0, TImagePtr const& image1)
{
  // auto adder = itk::AddImageFilter<TImage<TImagePtr>>::New();
  // adder->SetInput(image0);
  // adder->InPlaceOn();
  // adder->SetInput2(image1);
  // adder->Update();
  // image0 = adder->GetOutput();
  TImageIt<TImagePtr> iit0(image0, image0->GetRequestedRegion());
  TImageCIt<TImagePtr> iit1(image1, image1->GetRequestedRegion());
  while (!iit0.IsAtEnd()) {
    iit0.Value() += iit1.Value();
    ++iit0;
    ++iit1;
  }
}


// In-place: image *= multiplier
template <typename TImagePtr> void
multiplyImage (TImagePtr& image, double multiplier)
{
  auto filter = itk::MultiplyImageFilter<TImage<TImagePtr>>::New();
  filter->SetInput(image);
  filter->InPlaceOn();
  filter->SetConstant(multiplier);
  filter->Update();
  image = filter->GetOutput();
}


template <typename TImagePtr> TImagePtr
averageImages (std::vector<TImagePtr> const& images)
{
  TImagePtr ret = createImage<TImage<TImagePtr>>
      (images.front()->GetRequestedRegion(), 0.0);
  for (auto const& image: images) { addImage(ret, image); }
  multiplyImage(ret, 1.0 / images.size());
  return ret;
}


template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
skeletonizeImage (TImagePtrIn const& image)
{
  auto filter = itk::BinaryThinningImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  filter->SetInput(image);
  filter->Update();
  return filter->GetOutput();
}


template <typename TImagePtr> TImagePtr
cloneImage (TImagePtr const& image)
{
  auto duplicator = itk::ImageDuplicator<TImage<TImagePtr>>::New();
  duplicator->SetInputImage(image);
  duplicator->Update();
  return duplicator->GetOutput();
}


template <typename TImagePtr> void
copyImage (TImagePtr& dstImage, TImagePtr const& srcImage,
           typename TImage<TImagePtr>::RegionType const& srcRegion,
           typename TImage<TImagePtr>::IndexType const& dstIndex)
{
  typename TImage<TImagePtr>::RegionType dstRegion;
  dstRegion.SetIndex(dstIndex);
  dstRegion.SetSize(srcRegion.GetSize());
  TImageCIt<TImagePtr> sit(srcImage, srcRegion);
  TImageIt<TImagePtr> dit(dstImage, dstRegion);
  while (!dit.IsAtEnd()) {
    dit.Set(sit.Get());
    ++dit;
    ++sit;
  }
}


// Only works in 2D for now
template <typename TImagePtr> void
sampleImage (TImagePtr& outputImage, TImagePtr const& inputImage,
             std::vector<Int> const& offsets,
             std::vector<int> const& strides)
{
  const UInt D = TImage<TImagePtr>::ImageDimension;
  TImageLIIt<TImagePtr>
      iit(inputImage, inputImage->GetRequestedRegion()),
      oit(outputImage, outputImage->GetRequestedRegion());
  iit.SetDirection(0);
  oit.SetDirection(0);
  itk::Index<D> startIndex;
  for (int i = 0; i < D; ++i) { startIndex[i] = offsets[i]; }
  iit.SetIndex(startIndex);
  while (!oit.IsAtEnd()) {
    while (!oit.IsAtEndOfLine()) {
      oit.Value() = iit.Value();
      ++oit;
      for (int i = 0; i < strides[0]; ++i) { ++iit; }
    }
    oit.NextLine();
    for (int i = 0; i < strides[1]; ++i) { iit.NextLine(); }
  }
}


template <typename TImagePtr> TImagePtr
sampleImage (TImagePtr const& image, std::vector<Int> const& offsets,
             std::vector<int> const& strides)
{
  if (TImage<TImagePtr>::ImageDimension != 2)
  { perr("Error: image sampling only supports 2D..."); }
  auto reg = image->GetRequestedRegion();
  reg.GetModifiableIndex().Fill(0);
  for (int i = 0; i < TImage<TImagePtr>::ImageDimension; ++i) {
    reg.GetModifiableSize()[i] =
        std::ceil((reg.GetSize()[i] - offsets[i]) / (double)strides[i]);
  }
  auto ret = createImage<TImage<TImagePtr>>(reg);
  sampleImage(ret, image, offsets, strides);
  return ret;
}


// Generate boundary image (double-sized) of region segmentation
// Only works in 2D for now
// See BSDS/seg2bdry for reference
// f (TImageCLIIt<TImagePtr> const& viit0,
//    TImageCLIIt<TImagePtr> const& viit1)
template <typename TBImagePtr, typename TImagePtr, typename Func> void
genBoundaryImage (TBImagePtr& bImage, TImagePtr const& segImage,
                  Func f)
{
  bImage->FillBuffer(0);
  // Horizontal edges computed vertically
  TImageCLIIt<TImagePtr>
      viit0(segImage, segImage->GetRequestedRegion()),
      viit1(segImage, segImage->GetRequestedRegion());
  viit0.SetDirection(0);
  viit1.SetDirection(0);
  viit1.NextLine();
  while (!viit1.IsAtEnd()) {
    while (!viit1.IsAtEndOfLine()) {
      if (viit0.Get() != viit1.Get()) {
        auto index = viit0.GetIndex();
        index[0] = index[0] * 2 + 1;
        index[1] = index[1] * 2 + 2;
        bImage->SetPixel(index, f(viit0, viit1));
      }
      ++viit0;
      ++viit1;
    }
    viit0.NextLine();
    viit1.NextLine();
  }
  // Vertical edges computed horizontally
  TImageCLIIt<TImagePtr>
      hiit0(segImage, segImage->GetRequestedRegion()),
      hiit1(segImage, segImage->GetRequestedRegion());
  hiit0.SetDirection(1);
  hiit1.SetDirection(1);
  hiit1.NextLine();
  while (!hiit1.IsAtEnd()) {
    while (!hiit1.IsAtEndOfLine()) {
      if (hiit0.Get() != hiit1.Get()) {
        auto index = hiit0.GetIndex();
        index[0] = index[0] * 2 + 2;
        index[1] = index[1] * 2 + 1;
        bImage->SetPixel(index, f(hiit0, hiit1));
      }
      ++hiit0;
      ++hiit1;
    }
    hiit0.NextLine();
    hiit1.NextLine();
  }
  // Fill in non-border odd (x, y) boundary pixels
  TImageLIIt<TBImagePtr> oiit(bImage, bImage->GetRequestedRegion());
  oiit.SetDirection(0);
  oiit.NextLine();
  oiit.NextLine();
  ++(++oiit);
  TImageCLIIt<TBImagePtr>
      oiit0(bImage, bImage->GetRequestedRegion()),
      oiit1(bImage, bImage->GetRequestedRegion()),
      oiit2(bImage, bImage->GetRequestedRegion()),
      oiit3(bImage, bImage->GetRequestedRegion());
  oiit0.SetDirection(0);
  oiit1.SetDirection(0);
  oiit2.SetDirection(0);
  oiit3.SetDirection(0);
  oiit0.NextLine();
  ++(++oiit0);
  oiit1.NextLine();
  oiit1.NextLine();
  ++(++(++oiit1));
  oiit2.NextLine();
  oiit2.NextLine();
  oiit2.NextLine();
  ++(++oiit2);
  oiit3.NextLine();
  oiit3.NextLine();
  ++oiit3;
  while (!oiit2.IsAtEnd()) {
    while (!oiit1.IsAtEndOfLine()) {
      oiit.Set(std::max(std::max(oiit0.Get(), oiit2.Get()),
                        std::max(oiit1.Get(), oiit3.Get())));
      ++oiit1;
      if (oiit1.IsAtEndOfLine()) { break; }
      ++oiit1;
      ++(++oiit);
      ++(++oiit0);
      ++(++oiit2);
      ++(++oiit3);
    }
    oiit2.NextLine();
    if (oiit2.IsAtEnd()) { break; }
    oiit2.NextLine();
    oiit.NextLine();
    oiit.NextLine();
    oiit0.NextLine();
    oiit0.NextLine();
    oiit1.NextLine();
    oiit1.NextLine();
    oiit3.NextLine();
    oiit3.NextLine();
    ++(++oiit);
    ++(++oiit0);
    ++(++(++oiit1));
    ++(++oiit2);
    ++oiit3;
  }
  // Fill in border pixels
  auto width = getImageSize(bImage, 0);
  auto height = getImageSize(bImage, 1);
  auto _reg = bImage->GetRequestedRegion();
  auto _index = _reg.GetIndex();
  _reg.GetModifiableIndex()[1] = 1;
  _reg.GetModifiableSize()[1] = 1;
  copyImage(bImage, bImage, _reg, _index); // Top
  _reg.GetModifiableIndex()[1] = height - 2;
  _index[1] = height - 1;
  copyImage(bImage, bImage, _reg, _index); // Bottom
  _reg.GetModifiableIndex()[0] = 1;
  _reg.GetModifiableIndex()[1] = 0;
  _reg.GetModifiableSize()[0] = 1;
  _reg.GetModifiableSize()[1] = height;
  _index.Fill(0);
  copyImage(bImage, bImage, _reg, _index); // Left
  _reg.GetModifiableIndex()[0] = width - 2;
  _index[0] = width - 1;
  copyImage(bImage, bImage, _reg, _index); // Right
}


// Generate boundary image of region segmentation
// If doubleSize == false, use original size
// Only works in 2D for now
// See BSDS/seg2bdry for reference
// f (TImageCLIIt<TImagePtr> const& viit0,
//    TImageCLIIt<TImagePtr> const& viit1)
template <typename TImageOut, typename TImagePtr, typename Func>
typename TImageOut::Pointer
genBoundaryImage (TImagePtr const& segImage, Func f)
{
  if (TImage<TImagePtr>::ImageDimension != 2)
  { perr("Error: boundary image generation only supports 2D..."); }
  auto reg = segImage->GetRequestedRegion();
  reg.GetModifiableIndex().Fill(0);
  for (int i = 0; i < TImage<TImagePtr>::ImageDimension; ++i)
  { reg.GetModifiableSize()[i] = reg.GetSize()[i] * 2 + 1; }
  auto bImage = createImage<TImageOut>(reg);
  genBoundaryImage(bImage, segImage, f);
  return bImage;
}


// Relabel BG_VAL pixels to nearest smallest region
template <typename TImagePtr, typename TMaskPtr> void
dilateImage (TImagePtr& image, TMaskPtr const& mask)
{
  const uint D = TImage<TImagePtr>::ImageDimension;
  typedef TImageVal<TImagePtr> TKey;
  std::unordered_map<TKey, uint> cmap;
  genCountMap(cmap, image, mask);
  auto bgcit = cmap.find(BG_VAL);
  if (bgcit == cmap.end()) { return; }
  std::vector<itk::Index<D>> openSet;
  openSet.reserve(bgcit->second);
  std::vector<int> changeIndices;
  std::vector<TKey> changeKeys;
  changeIndices.reserve(bgcit->second);
  changeKeys.reserve(bgcit->second);
  cmap.erase(bgcit);
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (iit.Value() == BG_VAL) { openSet.push_back(iit.GetIndex()); }
  }
  std::vector<TKey> nvals;
  nvals.reserve(D * 2);
  int n = openSet.size();
  while (n > 0) {
    changeIndices.clear();
    changeKeys.clear();
    for (int i = 0; i < n; ++i) {
      nvals.clear();
      traverseNeighbors(openSet[i], image->GetRequestedRegion(), mask,
                        [&nvals, &image](itk::Index<D> const& idx)
                        {
                          auto val = image->GetPixel(idx);
                          if (val != BG_VAL) { nvals.push_back(val); }
                        });
      if (!nvals.empty()) {
        uint minSize = UINT_MAX;
        TKey minKey = BG_VAL;
        for (auto v: nvals) {
          auto sz = cmap.find(v)->second;
          if (sz < minSize) {
            minSize = sz;
            minKey = v;
          }
        }
        changeIndices.push_back(i);
        changeKeys.push_back(minKey);
      }
    }
    int m = changeIndices.size();
    for (int i = 0; i < m; ++i)
    { image->SetPixel(openSet[changeIndices[i]], changeKeys[i]); }
    remove(openSet, changeIndices);
    n = openSet.size();
  }
}


template <typename TImagePtr, typename TMaskPtr> RgbImage::Pointer
overlayImage (TImagePtr const& labelImage,
              typename UInt8Image<TImage<TImagePtr>::ImageDimension>
              ::Pointer const& bgImage, TMaskPtr const& mask,
              double opacity)
{
  const UInt D = TImage<TImagePtr>::ImageDimension;
  typedef itk::LabelImageToLabelMapFilter<TImage<TImagePtr>> LMF;
  auto lmf = LMF::New();
  lmf->SetInput(labelImage);
  auto lmof = itk::LabelMapOverlayImageFilter
      <typename LMF::OutputImageType, UInt8Image<D>, RgbImage>::New();
  lmof->SetInput(lmf->GetOutput());
  lmof->SetFeatureImage
      (bgImage.IsNotNull()? bgImage:
       createImage<UInt8Image<D>>(labelImage->GetRequestedRegion(), 255));
  lmof->SetOpacity(opacity);
  lmof->Update();
  return lmof->GetOutput();
}


template <typename T, typename TImagePtr, typename TMaskPtr> void
getImagePatches (std::vector<std::vector<T>>& patches,
                 TImagePtr const& image, TMaskPtr const& mask,
                 std::vector<int> const& patchRadius)
{
  const UInt D = TImage<TImagePtr>::ImageDimension;
  UInt n = getImageSize(image);
  patches.reserve(n);
  itk::Size<D> radius;
  uint np = 1;
  for (int i = 0; i < D; ++i) {
    radius[i] = patchRadius[i];
    np *= patchRadius[i] * 2 + 1;
  }
  for (itk::ConstNeighborhoodIterator<TImage<TImagePtr>>
           iit(radius, image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (mask.IsNull() ||
        mask->GetPixel(iit.GetIndex()) != MASK_OUT_VAL) {
      patches.emplace_back();
      patches.back().reserve(np);
      for (int i = 0; i < np; ++i)
      { patches.back().push_back(iit.GetPixel(i)); }
    }
  }
}

};

#endif
