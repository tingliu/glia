#ifndef _glia_type_region_hxx_
#define _glia_type_region_hxx_

#include "type/point_map.hxx"

namespace glia {

template <typename TKey, typename TPoint>
class TRegion : public TPointPtrMap<TKey, TPoint> {
 public:
  typedef TPointPtrMap<TKey, TPoint> Super;
  typedef TRegion<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef typename Super::Key Key;
  typedef typename Super::Point Point;
  typedef typename Super::Points Points;
  typedef TPointPtrMap<TKey, TPoint> Border;
  typedef TPointPtrPairMap<TKey, TPoint> Boundary;

  Border border;
  Boundary boundary;

  TRegion () {}

  TRegion (std::pair<TKey, Points*> const& pp) : Super(pp) {}

  TRegion (TKey key, Points* pPoints, Points* pBorder,
           std::vector<std::pair<TKey, Points*>> const& pBoundaries)
      : Super(std::make_pair(key, pPoints)) {
    if (pBorder) { border.emplace(key, pBorder); }
    for (auto const& bn: pBoundaries)
    { boundary.emplace(std::make_pair(key, bn.first), bn.second); }
  }

  ~TRegion () override {}

  virtual bool touchBorder () const { return !border.empty(); }

  // Get boundary on side of this region
  virtual void boundaryWith (Boundary& b, Self const& region) const {
    for (auto const& bp0: boundary) {
      for (auto const& bp1: region.boundary) {
        if (bp0.first.second == bp1.first.first) {
          b.merge(bp0);
          break;
        }
      }
    }
  }

  virtual void
  merge (TKey key, Points* pPoint, Points* pBorder,
         std::vector<std::pair<TKey, Points*>> const& pBoundaries) {
    Super::merge(std::make_pair(key, pPoint));
    border.merge(std::make_pair(key, pBorder));
    for (auto const& bn: pBoundaries) {
      auto it = boundary.find(std::make_pair(bn.first, key));
      if (it == boundary.end())
      { boundary[std::make_pair(key, bn.first)] = bn.second; }
      else { boundary.erase(it); }
    }
  }

  virtual void merge (Self const& region) {
    Super::merge(region);
    border.merge(region.border);
    for (auto const& pp: region.boundary) {
      auto it = boundary.find
          (std::make_pair(pp.first.second, pp.first.first));
      if (it == boundary.end()) { boundary.merge(pp); }
      else { boundary.erase(it); }
    }
  }

  // neighbors: keys of basic superpixels
  virtual void subregion
  (Self& sreg, std::unordered_set<TKey> const& neighbors) {
    static_cast<Super&>(sreg) = static_cast<Super&>(*this);
    sreg.border = this->border;
    for (auto const& bp: this->boundary) {
      if (neighbors.count(bp.first.second) > 0)
      { sreg.boundary.insert(bp); }
    }
  }
};

};

#endif
