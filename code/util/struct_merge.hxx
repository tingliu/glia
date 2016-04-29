#ifndef _glia_util_struct_merge_hxx_
#define _glia_util_struct_merge_hxx_

#include "type/boundary_table.hxx"
#include "type/tuple.hxx"
#include "util/stats.hxx"

namespace glia {

template <typename TBTItemData, typename TRegionMap, typename IFb,
          typename UFb, typename IFsal, typename UFsal,
          typename CFunc> void
genMergeOrderGreedy
(std::vector<TTriple<typename TRegionMap::Key>>& order,
 std::vector<double>& saliencies, TRegionMap& rmap, bool updateRegion,
 IFb initFb, IFsal initFsal, UFb updateFb, UFsal updateFsal, CFunc fcond)
{
  TBoundaryTable<TBTItemData, TRegionMap> bt(rmap, initFb, initFsal);
  typename TRegionMap::Key keyToAssign = rmap.maxKey() + 1;
  order.reserve(order.size() + rmap.size() - 1);
  saliencies.reserve(saliencies.size() + rmap.size() - 1);
  while (!bt.empty()) {
    auto btit = bt.top(fcond);
    if (btit == bt.table().end()) { break; } // Stop if no item satifies
    auto r0 = btit->first.first;
    auto r1 = btit->first.second;
    order.push_back
        (TTriple<typename TRegionMap::Key>(r0, r1, keyToAssign));
    saliencies.push_back(btit->second->mqit->first);
    if (updateRegion) { rmap.merge(r0, r1, keyToAssign); }
    bt.update(btit, keyToAssign++, updateFb, updateFsal);
  }
}


template <typename TRegionMap, typename TImagePtr,
          typename CFunc> void
genMergeOrderGreedyUsingPbMean
(std::vector<TTriple<typename TRegionMap::Key>>& order,
 std::vector<double>& saliencies, TRegionMap& rmap, bool updateRegion,
 TImagePtr const& pbImage, CFunc fcond)
{
  typedef std::pair<double, int> ItemData;
  typedef typename TRegionMap::Key Key;
  auto initFb = [&pbImage, &rmap](ItemData& data, Key r0, Key r1)
      {
        auto rit0 = rmap.find(r0);
        auto rit1 = rmap.find(r1);
        typename TRegionMap::Region::Boundary b;
        getBoundary(b, rit0->second, rit1->second);
        data.first = 0.0;
        b.traverse([&pbImage, &data](typename TRegionMap::Point const& p){
            data.first += pbImage->GetPixel(p); });
        data.second = b.size();
        data.first = sdivide(data.first, data.second, 0.0);
      };
  auto initFsal = [](ItemData& data, Key r0, Key r1) -> double {
    if (data.first == DUMMY)
    { perr("Error: invalid boundary saliency..."); }
    return -data.first;
  };
  auto updateFb = [](ItemData& data2s, Key r0, Key r1, Key rs, Key r2,
                     ItemData* pData0s, ItemData* pData1s)
      {
        data2s.first = 0.0;
        data2s.second = 0;
        if (pData0s) {
          data2s.first += pData0s->first * pData0s->second;
          data2s.second += pData0s->second;
        }
        if (pData1s) {
          data2s.first += pData1s->first * pData1s->second;
          data2s.second += pData1s->second;
        }
        data2s.first = sdivide(data2s.first, data2s.second, 0.0);
      };
  auto updateFsal = [](ItemData& data2s, Key rs, Key r2) -> double {
    if (data2s.first == DUMMY)
    { perr("Error: invalid boundary saliency..."); }
    return -data2s.first;
  };
  genMergeOrderGreedy<ItemData>
      (order, saliencies, rmap, updateRegion, initFb, initFsal,
       updateFb, updateFsal, fcond);
}


template <typename TRegionMap, typename TImagePtr,
          typename CFunc, typename AFunc> void
genMergeOrderGreedyUsingPbApproxMedian
(std::vector<TTriple<typename TRegionMap::Key>>& order,
 std::vector<double>& saliencies, TRegionMap& rmap, bool updateRegion,
 TImagePtr const& pbImage, CFunc fcond, AFunc faux)
{
  typedef std::vector<double> ItemData;
  typedef typename TRegionMap::Key Key;
  // Saliency and item data update functions
  auto initFb =
      [&pbImage, &rmap, &faux](ItemData& data, Key r0, Key r1)
      {
        auto rit0 = rmap.find(r0);
        auto rit1 = rmap.find(r1);
        typename TRegionMap::Region::Boundary b;
        getBoundary(b, rit0->second, rit1->second);
        data.reserve(b.size());
        b.traverse(
            [&pbImage, &data](typename TRegionMap::Point const& p) {
              data.push_back(pbImage->GetPixel(p));
            });
        faux(data, r0, r1);
      };
  auto initFsal = [](ItemData& data, Key r0, Key r1) -> double
      {
        double p = stats::amedian(data);
        if (p == DUMMY) { perr("Error: invalid boundary saliency..."); }
        return -p;
      };
  auto updateFb =
      [&faux](ItemData& data2s, Key r0, Key r1, Key rs, Key r2,
         ItemData* pData0s, ItemData* pData1s)
      {
        if (pData0s && pData1s) { splice(data2s, *pData0s, *pData1s); }
        else if (pData0s) { splice(data2s, *pData0s); }
        else if (pData1s) { splice(data2s, *pData1s); }
        faux(data2s, rs, r2);
      };
  auto updateFsal = [](ItemData& data2s, Key rs, Key r2) -> double
      {
        double p = stats::amedian(data2s);
        if (p == DUMMY) { perr("Error: invalid boundary saliency..."); }
        return -p;
      };
  genMergeOrderGreedy<ItemData>
      (order, saliencies, rmap, updateRegion, initFb, initFsal,
       updateFb, updateFsal, fcond);
}


template <typename TRegionMap, typename TImagePtr,
          typename CFunc> void
genMergeOrderGreedyUsingPbApproxMedianAndMinSize
(std::vector<TTriple<typename TRegionMap::Key>>& order,
 std::vector<double>& saliencies, TRegionMap& rmap,
 TImagePtr const& pbImage, CFunc fcond)
{
  typedef std::vector<double> ItemData;
  typedef typename TRegionMap::Key Key;
  // Saliency and item data update functions
  auto initFb =
      [&pbImage, &rmap](ItemData& data, Key r0, Key r1)
      {
        auto rit0 = rmap.find(r0);
        auto rit1 = rmap.find(r1);
        typename TRegionMap::Region::Boundary b;
        getBoundary(b, rit0->second, rit1->second);
        data.reserve(b.size());
        b.traverse([&pbImage, &data](typename TRegionMap::Point const& p)
  { data.push_back(pbImage->GetPixel(p)); });
      };
  auto initFsal = [&rmap](ItemData& data, Key r0, Key r1) -> double
      {
        double p = stats::amedian(data);
        if (p == DUMMY) { perr("Error: invalid boundary saliency..."); }
        return -p * std::min(rmap.find(r0)->second.size(),
                             rmap.find(r1)->second.size());
      };
  auto updateFb =
      [](ItemData& data2s, Key r0, Key r1, Key rs, Key r2,
         ItemData* pData0s, ItemData* pData1s)
      {
        if (pData0s && pData1s) { splice(data2s, *pData0s, *pData1s); }
        else if (pData0s) { splice(data2s, *pData0s); }
        else if (pData1s) { splice(data2s, *pData1s); }
      };
  auto updateFsal = [&rmap](ItemData& data2s, Key rs, Key r2) -> double
      {
        double p = stats::amedian(data2s);
        if (p == DUMMY) { perr("Error: invalid boundary saliency..."); }
        return -p * std::min(rmap.find(rs)->second.size(),
                             rmap.find(r2)->second.size());
      };
  genMergeOrderGreedy<ItemData>
      (order, saliencies, rmap, true, initFb, initFsal,
       updateFb, updateFsal, fcond);
}


template <typename TKey> void
transformKeys (std::unordered_map<TKey, TKey>& lmap,
               std::vector<TTriple<TKey>> const& order)
{
  std::unordered_set<TKey> newKeys;
  std::unordered_map<TKey, TKey> omap;
  for (auto const& merge: order) {
    omap[merge.x0] = merge.x2;
    omap[merge.x1] = merge.x2;
    newKeys.insert(merge.x2);
  }
  for (auto const& op: omap) {
    if (newKeys.count(op.first) == 0) {
      TKey dst = op.second;
      auto oit = omap.find(dst);
      while (oit != omap.end()) {
        dst = oit->second;
        oit = omap.find(dst);
      }
      lmap[op.first] = dst;
    }
  }
}


template <typename TKey> void
getBaseKeys (std::unordered_set<TKey>& baseKeys,
             std::vector<TTriple<TKey>> const& order)
{
  std::unordered_set<TKey> newKeys;
  for (auto const& merge: order) {
    if (newKeys.count(merge.x0) == 0) { baseKeys.insert(merge.x0); }
    if (newKeys.count(merge.x1) == 0) { baseKeys.insert(merge.x1); }
    newKeys.insert(merge.x2);
  }
}

};

#endif
