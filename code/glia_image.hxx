#ifndef _glia_image_hxx_
#define _glia_image_hxx_

#include "glia_base.hxx"
#include "itkSmartPointer.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkRGBPixel.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkLineIterator.h"
#include "itkLineConstIterator.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"

namespace glia {

typedef itk::IndexValueType Int;
typedef itk::SizeValueType UInt;

const Label BG_VAL = 0;
const Label MASK_OUT_VAL = 0;
const Label MASK_IN_VAL = MASK_OUT_VAL + 1;

typedef itk::RGBPixel<uint8> Rgb;
typedef itk::Image<Rgb, 2> RgbImage;

template <UInt D> using BoolImage = itk::Image<bool, D>;
template <UInt D> using UInt8Image = itk::Image<uint8, D>;
template <UInt D> using UInt16Image = itk::Image<uint16, D>;
template <UInt D> using IntImage = itk::Image<int, D>;
template <UInt D> using LabelImage = itk::Image<Label, D>;
template <UInt D> using RealImage = itk::Image<Real, D>;
template <UInt D> using RealVecImage = itk::VectorImage<Real, D>;

template <typename TImagePtr> using TImage =
    typename TImagePtr::ObjectType;

template <typename TImagePtr> using TImageVal =
    typename TImagePtr::ObjectType::PixelType;

template <typename TVecImagePtr> using TVecImageVal =
    typename TVecImagePtr::ObjectType::PixelType::ValueType;

template <typename TImagePtr> using TImageIt =
    typename itk::ImageRegionIterator<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageCIt =
    typename itk::ImageRegionConstIterator<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageIIt =
    typename itk::ImageRegionIteratorWithIndex<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageCIIt =
    typename itk::ImageRegionConstIteratorWithIndex<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageLIIt =
    typename itk::ImageLinearIteratorWithIndex<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageCLIIt =
    typename itk::ImageLinearConstIteratorWithIndex<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageSIIt =
    typename itk::ImageSliceIteratorWithIndex<TImage<TImagePtr>>;

template <typename TImagePtr> using TImageCSIIt =
    typename itk::ImageSliceConstIteratorWithIndex<TImage<TImagePtr>>;

};

#endif
