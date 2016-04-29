#ifndef _glia_type_neighbor_hxx_
#define _glia_type_neighbor_hxx_

#include "type/point.hxx"

namespace glia {

// border.previous.i: [i * 2]
// border.next.i: [i * 2 + 1]
// boundary.previous.i: [i * 2 + 8]
// boundary.next.i: [i * 2 + 9]

class NeighborIndicator : public Object, public std::bitset<16> {
 public:
  typedef Object SuperObject;
  typedef std::bitset<16> Super;
  typedef NeighborIndicator Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef Self* WeakPointer;

  NeighborIndicator () {}

  template <typename TImagePtr, typename TMaskPtr>
  NeighborIndicator
  (itk::Index<TImage<TImagePtr>::ImageDimension> const& point,
   TImagePtr const& image, TMaskPtr const& mask)
  { set(point, image, mask); }

  ~NeighborIndicator () override {}

  template <typename TImagePtr, typename TMaskPtr> void
  set (itk::Index<TImage<TImagePtr>::ImageDimension> point,
       TImagePtr const& image, TMaskPtr const& mask) {
    auto const& region = image->GetRequestedRegion();
    auto val = image->GetPixel(point);
    for (UInt i = 0; i < TImage<TImagePtr>::ImageDimension; ++i) {
      --point[i];
      if (!region.IsInside(point) ||
          (mask.IsNotNull() && mask->GetPixel(point) == MASK_OUT_VAL))
      { Super::operator[](i << 1) = 1; }
      else if (image->GetPixel(point) != val)
      { Super::operator[]((i << 1) + 8) = 1; }
      point[i] += 2;
      if (!region.IsInside(point) ||
          (mask.IsNotNull() && mask->GetPixel(point) == MASK_OUT_VAL))
      { Super::operator[]((i << 1) + 1) = 1; }
      else if (image->GetPixel(point) != val)
      { Super::operator[]((i << 1) + 9) = 1; }
      --point[i];
    }
  }

  virtual bool isOnBorder () const
  { return (Super::to_ulong() & 255ul) != 0; }

  virtual bool isOnBoundary () const
  { return Super::to_ulong() > 255ul; }
};


template <typename TImagePtr, typename TMaskPtr> inline bool
isPointOnBorder
(itk::Index<TImage<TImagePtr>::ImageDimension> const& point,
 TImagePtr const& image, TMaskPtr const& mask)
{
  NeighborIndicator ni(point, image, mask);
  return ni.isOnBorder();
}


template <typename TImageRegion, typename TMaskPtr, typename Func>
inline void
traverseNeighbors (itk::Index<TImageRegion::ImageDimension> point,
                   TImageRegion const& region, TMaskPtr const& mask,
                   Func f)
{
  for (UInt i = 0; i < TImageRegion::ImageDimension; ++i) {
    --point[i];
    if (region.IsInside(point) &&
        (mask.IsNull() || mask->GetPixel(point) != MASK_OUT_VAL))
    { f(point); }
    point[i] += 2;
    if (region.IsInside(point) &&
        (mask.IsNull() || mask->GetPixel(point) != MASK_OUT_VAL))
    { f(point); }
    --point[i];
  }
}


template <typename TContainer, typename TImagePtr,
          typename TMaskPtr> inline void
getNeighborValues(TContainer& nvs,
                  itk::Index<TImage<TImagePtr>::ImageDimension> point,
                  TImagePtr const& image, TMaskPtr const& mask)
{
  nvs.reserve(TImage<TImagePtr>::ImageDimension << 1);
  traverseNeighbors
      (point, image->GetRequestedRegion(), mask, [&nvs, &image]
       (Point<TImage<TImagePtr>::ImageDimension> point)
       { nvs.insert(nvs.end(), image->GetPixel(point)); });
}


// Returns (neighbor value, is on border)
// If point does not have a unique neighbor different than this value
// Neighbor value = this value
template <typename TImagePtr, typename TMaskPtr> inline
std::pair<TImageVal<TImagePtr>, bool>
getContourTraits (itk::Index<TImage<TImagePtr>::ImageDimension> point,
                  TImagePtr const& image, TMaskPtr const& mask)
{
  std::vector<TImageVal<TImagePtr>> nvs;
  getNeighborValues(nvs, point, image, mask);
  auto thisVal = image->GetPixel(point);
  auto ret = std::make_pair
      (thisVal, nvs.size() < (TImage<TImagePtr>::ImageDimension << 1));
  for (auto x: nvs) {
    if (x != thisVal) {
      ret.first = x;
      break;
    }
  }
  return ret;
}


};

#endif
