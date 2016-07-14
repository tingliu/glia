#ifndef _glia_util_struct_hxx_
#define _glia_util_struct_hxx_

#include "type/hash.hxx"
#include "type/neighbor.hxx"
#include "util/container.hxx"

namespace glia {

template <typename TRegion> inline void
getBoundary (typename TRegion::Boundary& b01, TRegion const& region0,
             TRegion const& region1)
{
  region0.boundaryWith(b01, region1);
  region1.boundaryWith(b01, region0);
}


template <typename TImagePtr, typename TMaskPtr> uint
countPoints (TImagePtr const& image, TMaskPtr const& mask,
             TImageVal<TImagePtr> const& val)
{
  uint n = 0;
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (iit.Value() == val &&
        (mask.IsNull() ||
         mask->GetPixel(iit.GetIndex()) != MASK_OUT_VAL)) { ++n; }
  }
  return n;
}


template <typename TPoints, typename TImagePtr, typename TMaskPtr> void
getPoints (TPoints& points, TImagePtr const& image,
           TMaskPtr const& mask, TImageVal<TImagePtr> const& val,
           uint nPoints)
{
  points.reserve(nPoints);
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    auto& index = iit.GetIndex();
    if (iit.Value() == val &&
        (mask.IsNull() || mask->GetPixel(index) != MASK_OUT_VAL))
    { points.emplace_back(index); }
  }
}


template <typename TSet, typename TImagePtr, typename TMaskPtr> void
getKeys (TSet& keys, TImagePtr const& image, TMaskPtr const& mask)
{
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    if (mask.IsNull() || mask->GetPixel(iit.GetIndex()) != MASK_OUT_VAL)
    { keys.insert(iit.Value()); }
  }
}


template <typename TCMap, typename TImagePtr, typename TMaskPtr> void
genCountMap (TCMap& cmap, TImagePtr const& image, TMaskPtr const& mask)
{
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    auto& index = iit.GetIndex();
    if (mask.IsNull() || mask->GetPixel(index) != MASK_OUT_VAL) {
      auto key = iit.Value();
      auto cit = cmap.find(key);
      if (cit == cmap.end()) { cmap[key] = 1; }
      else { ++cit->second; }
    }
  }
}


template <typename TPMap, typename TImagePtr, typename TMaskPtr> void
genPointMap (TPMap& pmap, TImagePtr const& image, TMaskPtr mask)
{
  std::unordered_map<typename TPMap::key_type, uint> cmap;
  genCountMap(cmap, image, mask);
  for (auto const& cp: cmap) {
    auto& p = pmap[cp.first];
    p.reserve(cp.second);
  }
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    auto& index = iit.GetIndex();
    if (mask.IsNull() || mask->GetPixel(index) != MASK_OUT_VAL)
    { pmap[iit.Value()].push_back(index); }
  }
}


template <typename TPoints, typename TBMap, typename TImagePtr,
          typename TMaskPtr> void
getContour (TPoints& border, TBMap& boundaryMap, TPoints const& points,
            TImagePtr const& image, TMaskPtr const& mask)
{
  for (auto const& p: points) {
    auto thisVal = image->GetPixel(p);
    auto ct = getContourTraits(p, image, mask);
    if (ct.first != thisVal) // boundary point
    { boundaryMap[std::make_pair(thisVal, ct.first)].push_back(p); }
    else if (ct.second) // border point
    { border.push_back(p); }
  }
}


// Generate border/boundary map using point map
template <typename TPMap, typename TBMap, typename TImagePtr,
          typename TMaskPtr> void
genContourMap (TPMap& borderMap, TBMap& boundaryMap,
               TPMap const& pointMap, TImagePtr const& image,
               TMaskPtr const& mask)
{
  for (auto const& pp: pointMap) {
    auto it = borderMap.insert
        (std::make_pair(pp.first, typename TPMap::mapped_type())).first;
    getContour(it->second, boundaryMap, pp.second, image, mask);
    if (it->second.empty()) { borderMap.erase(it); }
  }
}


// Generate border/boundary map using image/mask
template <typename TPMap, typename TBMap, typename TImagePtr,
          typename TMaskPtr> void
genContourMap (TPMap& borderMap, TBMap& boundaryMap,
               TImagePtr const& image, TMaskPtr const& mask)
{
  for (TImageCIIt<TImagePtr> iit(image, image->GetRequestedRegion());
       !iit.IsAtEnd(); ++iit) {
    Point<TImage<TImagePtr>::ImageDimension> point(iit.GetIndex());
    auto thisVal = image->GetPixel(point);
    auto ct = getContourTraits(point, image, mask);
    if (ct.first != thisVal) { // boundary point
      boundaryMap[std::make_pair(thisVal, ct.first)].push_back(point);
    }
    else if (ct.second) // border point
    { borderMap[thisVal].push_back(point); }
  }
}


template <typename TRMap> void
groupRegions
(std::vector<std::unordered_set<typename TRMap::Key>>& groups,
 TRMap const& rmap)
{
  typedef typename TRMap::Key Key;
  std::unordered_map<Key, int> gmap; // region key -> group index
  uint n = groups.size();
  for (int i = 0; i < n; ++i)
  { for (auto const& key: groups[i]) { gmap[key] = i; } }
  std::set<int> indicesToRemove;
  std::vector<Key> curGroup, newGroup;
  for (int i = 0; i < n; ++i) {
    newGroup.clear();
    newGroup.reserve(groups[i].size());
    for (auto const& key: groups[i]) { newGroup.push_back(key); }
    while (!newGroup.empty()) {
      curGroup.clear();
      curGroup.swap(newGroup);
      for (auto const& key0: curGroup) {
        for (auto const& bp: rmap.find(key0)->second.boundary) {
          auto git = gmap.find(bp.first.second);
          if (git != gmap.end() && git->second != i) {
            int oldIndex = git->second;
            for (auto const& key1: groups[oldIndex]) {
              newGroup.push_back(key1);
              gmap[key1] = i;
            }
            groups[oldIndex].clear();
            indicesToRemove.insert(oldIndex);
          }
        }
      }
      for (auto const& key: newGroup) { groups[i].insert(key); }
    }
  }
  remove(groups, indicesToRemove);
}


template <typename TSKey, typename TRKey> void
groupRegions (
    std::list<std::list<std::pair<TSKey, TRKey>>>& groups,
    std::unordered_set<std::pair<TSKey, TRKey>> const& regions,
    std::vector<
    std::pair<std::pair<TSKey, TRKey>, std::pair<TSKey, TRKey>>>
    const& links)
{
  std::unordered_map<
    std::pair<TSKey, TRKey>,
    typename std::list<std::list<std::pair<TSKey, TRKey>>>::iterator>
      gmap;
  for (auto const& r : regions) {
    auto it = groups.emplace(groups.end());
    it->push_back(r);
    gmap[r] = it;
  }
  // // Debug
  // int i = 0;
  // for (auto const& g : groups) {
  //   std::cout << "group " << i++ << ": ";
  //   for (auto const& r : g) {
  //     std::cout << "(" << r.first << ", " << r.second << ") ";
  //   }
  //   std::cout << std::endl;
  // }
  // for (auto const& g : gmap) {
  //   std::cout << "gmap[(" << g.first.first << ", "
  //             << g.first.second << ")] = {";
  //   for (auto const& r : *g.second) {
  //     std::cout << "(" << r.first << ", " << r.second << ") ";
  //   }
  //   std::cout << "}" << std::endl;
  // }
  // std::cout << std::endl;
  // // ~ Debug
  for (auto const& link : links) {
    auto git0 = gmap.find(link.first)->second;
    auto git1 = gmap.find(link.second)->second;
    auto& g0 = *git0;
    auto& g1 = *git1;
    if (git0 != git1) {
      for (auto const& r : g1) { gmap[r] = git0; }
      g0.splice(g0.end(), g1);
      groups.erase(git1);
    }
    // // Debug
    // std::cout << "link = [(" << link.first.first << ", "
    //           << link.first.second << ") -- ("
    //           << link.second.first << ", "
    //           << link.second.second << ")]" << std::endl;
    // i = 0;
    // for (auto const& g : groups) {
    //   std::cout << "group " << i++ << ": ";
    //   for (auto const& r : g) {
    //     std::cout << "(" << r.first << ", " << r.second << ") ";
    //   }
    //   std::cout << std::endl;
    // }
    // for (auto const& g : gmap) {
    //   std::cout << "gmap[(" << g.first.first << ", "
    //             << g.first.second << ")] = {";
    //   for (auto const& r : *g.second) {
    //     std::cout << "(" << r.first << ", " << r.second << ") ";
    //   }
    //   std::cout << "}" << std::endl;
    // }
    // std::cout << std::endl;
    // // ~ Debug
  }
}

};

#endif
