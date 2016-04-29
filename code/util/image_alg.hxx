#ifndef _glia_util_image_alg_hxx_
#define _glia_util_image_alg_hxx_

#include "util/image.hxx"
#include "itkMorphologicalWatershedImageFilter.h"

namespace glia {

template <typename TImageOut, typename TImagePtrIn>
typename TImageOut::Pointer
watershed (TImagePtrIn const& inputImage, double level)
{
  const UInt D = TImage<TImagePtrIn>::ImageDimension;
  auto ws = itk::MorphologicalWatershedImageFilter
      <TImage<TImagePtrIn>, TImageOut>::New();
  ws->SetInput(inputImage);
  ws->SetLevel(level);
  ws->MarkWatershedLineOff();
  ws->Update();
  return ws->GetOutput();
}

};

#endif
