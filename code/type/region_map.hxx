#ifndef _glia_type_region_map_hxx_
#define _glia_type_region_map_hxx_

#include "type/region.hxx"
#include "util/struct.hxx"
#include "util/container.hxx"
#include "type/tuple.hxx"

namespace glia {

template <typename TKey, typename TPoint>
class TRegionMap
    : public Object,
      public std::unordered_map<TKey, TRegion<TKey, TPoint>> {
public:
  typedef Object SuperObject;
  typedef std::unordered_map<TKey, TRegion<TKey, TPoint>> Super;
  typedef TRegionMap<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef typename Super::iterator iterator;
  typedef typename Super::const_iterator const_iterator;
  typedef TKey Key;
  typedef TRegion<TKey, TPoint> Region;
  typedef typename Region::Point Point;
  typedef typename Region::Points Points;
  typedef TPointMap<TKey, TPoint> PointMap;
  typedef TPointPairMap<TKey, TPoint> PointPairMap;

  std::shared_ptr<PointMap> pPointMap = std::make_shared<PointMap>();
  std::shared_ptr<PointMap> pBorderMap = std::make_shared<PointMap>();
  std::shared_ptr<PointPairMap> pBoundaryMap =
      std::make_shared<PointPairMap>();

  TRegionMap () {}

  template <typename TImagePtr, typename TMaskPtr>
      TRegionMap (TImagePtr const& image, TMaskPtr const& mask,
                  bool onlyContour) { set(image, mask, onlyContour); }

  template <typename TImagePtr, typename TMaskPtr>
      TRegionMap (TImagePtr const& image, TMaskPtr const& mask,
                  std::vector<TTriple<TKey>> const& order,
                  bool onlyContour) {
    set(image, mask, onlyContour);
    set(order);
  }

  ~TRegionMap () override {}

  template <typename TImagePtr, typename TMaskPtr> void
      set (TImagePtr const& image, TMaskPtr const& mask,
           bool onlyContour) {
    if (onlyContour) {
      genContourMap(*pBorderMap, *pBoundaryMap, image, mask);
      initContour();
    }
    else {
      genPointMap(*pPointMap, image, mask);
      genContourMap
          (*pBorderMap, *pBoundaryMap, *pPointMap, image, mask);
      init();
    }
  }

  void set (std::vector<TTriple<TKey>> const& order)
  { for (auto const& m: order) { merge(m.x0, m.x1, m.x2); } }

  virtual Key maxKey () const {
    TKey ret = Super::begin()->first;
    for (auto const& rp: *this)
    { if (ret < rp.first) { ret = rp.first; } }
    return ret;
  }

  // Initialize region points/borders/boundaries
  // Valid after pointMap/borderMap/boundaryMap initialized
  virtual void init () {
    Super::clear();
    pBoundaryMap->prepare();
    for (auto& pp: *pPointMap) {
      auto it = Super::emplace
          (pp.first, std::make_pair(pp.first, &pp.second)).first;
      auto pBorder = cpointer(*pBorderMap, pp.first);
      if (pBorder) { it->second.border[pp.first] = pBorder; }
      auto bnit = pBoundaryMap->find(pp.first);
      if (bnit != pBoundaryMap->unary().end()) {
        for (auto const& bp: bnit->second) {
          it->second.boundary.emplace
              (std::make_pair(pp.first, bp.first), bp.second);
        }
      }
    }
  }

  // Initialize region borders/boundaries
  // Valid after borderMap/boundaryMap initialized
  virtual void initContour () {
    Super::clear();
    pBoundaryMap->prepare();
    for (auto& pp: pBoundaryMap->unary()) {
      auto it = Super::insert(std::make_pair(pp.first, Region())).first;
      auto pBorder = cpointer(*pBorderMap, pp.first);
      if (pBorder) { it->second.border[pp.first] = pBorder; }
      for (auto const& bp: pp.second) {
        it->second.boundary.emplace
            (std::make_pair(pp.first, bp.first), bp.second);
      }
    }
  }

  virtual iterator merge (TKey r0, TKey r1, TKey r2) {
    auto it = citerator(*static_cast<Super*>(this), r2);
    if (r2 != r0) { it->second.merge(Super::find(r0)->second); }
    if (r2 != r1) { it->second.merge(Super::find(r1)->second); }
    return it;
  }

  virtual void subregions
      (Self& sregs, std::unordered_set<TKey> const& keys) {
    std::unordered_set<TKey> neighbors;
    for (auto const& key: keys) {
      for (auto const& bp: Super::find(key)->second.boundary)
      { neighbors.insert(bp.first.first); }
    }
    sregs.pPointMap = pPointMap;
    sregs.pBorderMap = pBorderMap;
    sregs.pBoundaryMap = pBoundaryMap;
    for (auto const& key: keys)
    { Super::find(key)->second.subregion(sregs[key], neighbors); }
  }
};

};

#endif
