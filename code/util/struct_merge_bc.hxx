#ifndef _glia_util_struct_merge_bc_hxx_
#define _glia_util_struct_merge_bc_hxx_

#include "util/struct_merge.hxx"

namespace glia {

template <typename TBcFeat, typename TRegionMap, typename FFunc,
          typename BCFunc, typename CFunc> void
genMergeOrderGreedyUsingBoundaryClassifier (
    std::vector<TTriple<typename TRegionMap::Key>>& order,
    std::vector<double>& saliencies, TRegionMap& rmap,
    FFunc fBcFeat, BCFunc fBcPred, CFunc fcond)
{
  typedef TBcFeat ItemData;
  typedef TBoundaryTable<ItemData, TRegionMap> BoundaryTable;
  typedef typename TRegionMap::Key Key;
  auto initFb = [&rmap, &fBcFeat](ItemData& data, Key r0, Key r1) {
    rmap.erase(BG_VAL);
    auto rit2 = rmap.merge(r0, r1, BG_VAL);
    fBcFeat(data, rmap.find(r0)->second, rmap.find(r1)->second,
            rit2->second, r0, r1, BG_VAL);
  };
  auto initFsal = [&fBcPred](
      ItemData const& data, Key r0, Key r1) -> double {
    return fBcPred(data);
  };
  auto updateFb = [&rmap, &fBcFeat](
      ItemData& data2s, Key r0, Key r1, Key rs, Key r2,
      ItemData* pData0s, ItemData* pData1s) {
    rmap.erase(BG_VAL);
    auto rit3 = rmap.merge(rs, r2, BG_VAL);
    fBcFeat(data2s, rmap.find(rs)->second, rmap.find(r2)->second,
            rit3->second, rs, r2, BG_VAL);
  };
  auto updateFsal = [&fBcPred](
      ItemData const& data2s, Key rs, Key r2) -> double {
    return fBcPred(data2s);
  };
  genMergeOrderGreedy<ItemData>(
      order, saliencies, rmap, true, initFb, initFsal,
      updateFb, updateFsal, fcond);
}


template <typename TBcFeat, typename TKey, typename TSegImagePtr,
          typename TMaskPtr, typename FFunc, typename BCFunc,
          typename CFunc> void
genMergeOrderGreedyUsingBoundaryClassifier (
    std::vector<TTriple<TKey>>& order, std::vector<double>& saliencies,
    TSegImagePtr const& segImage, TMaskPtr const& mask,
    FFunc fBcFeat, BCFunc fBcPred, CFunc fcond)
{
  TRegionMap<TKey, Point<TImage<TSegImagePtr>::ImageDimension>>
      rmap(segImage, mask, false); // Both region points and contours
  genMergeOrderGreedyUsingBoundaryClassifier<TBcFeat>(
      order, saliencies, rmap, fBcFeat, fBcPred, fcond);
}

};

#endif
