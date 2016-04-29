#ifndef _glia_hmt_bc_label_hxx_
#define _glia_hmt_bc_label_hxx_

#include "util/image_stats.hxx"

namespace glia {
namespace hmt {

const int BC_LABEL_UNKNOWN = 0;  // Unknown: 0
const int BC_LABEL_SPLIT = 1;    // Split: +1
const int BC_LABEL_MERGE = -1;   // Merge: -1
const int RC_LABEL_UNKNOWN = 0;  // Unknown: 0
const int RC_LABEL_TRUE = 1;     // True node: +1
const int RC_LABEL_FALSE = -1;   // False node: -1

template <typename TRegion, typename TImagePtr> int
genBoundaryClassificationLabelVI (
    double& mergeVI, double& splitVI, TRegion const& region0,
    TRegion const& region1, TRegion const& region2,
    TImagePtr const& truthImage)
{
  std::vector<TRegion const*> pSplitRegions{&region0, &region1};
  std::vector<TRegion const*> pMergeRegions{&region2};
  mergeVI = stats::vi(pMergeRegions, truthImage, {BG_VAL});
  splitVI = stats::vi(pSplitRegions, truthImage, {BG_VAL});
  return mergeVI < splitVI? BC_LABEL_MERGE: BC_LABEL_SPLIT;
}


template <typename TRegionMap, typename TImagePtr> int
genBoundaryClassificationLabelVI
(double& mergeVI, double& splitVI, TRegionMap const& rmap,
 typename TRegionMap::Key r0, typename TRegionMap::Key r1,
 typename TRegionMap::Key r2, TImagePtr const& truthImage)
{
  return genBoundaryClassificationLabelVI(
      mergeVI, splitVI, rmap.find(r0)->second, rmap.find(r1)->second,
      rmap.find(r2)->second, truthImage);
}


// Decide labels based on pair F1
// Use maxPrecDrop >= 1.0 to disable
template <typename TInt, typename TRegion, typename TImagePtr> int
genBoundaryClassificationLabelF1 (
    double& mergeF1, double& splitF1,
    std::vector<TRegion const*> const& pSplitRegions,
    std::vector<TRegion const*> const& pMergeRegions,
    TImagePtr const& truthImage, bool tweak, double maxPrecDrop)
{
  double splitPrec, splitRec, mergePrec, mergeRec;
  stats::pairF1<TInt>(splitF1, splitPrec, splitRec, pSplitRegions,
                      truthImage, {BG_VAL});
  stats::pairF1<TInt>(mergeF1, mergePrec, mergeRec, pMergeRegions,
                      truthImage, {BG_VAL});
  if (maxPrecDrop < 1.0 && splitPrec - mergePrec > maxPrecDrop)
  { return BC_LABEL_SPLIT; }
  if (tweak) {
    return (mergeF1 > splitF1  ||
            (splitPrec < FEPS && splitRec < FEPS &&
             mergePrec < FEPS && mergeRec < FEPS) ||
            (splitF1 == mergeF1 && splitPrec > 0.9 && mergePrec > 0.9)?
            BC_LABEL_MERGE: BC_LABEL_SPLIT);
  }
  return mergeF1 > splitF1 ? BC_LABEL_MERGE: BC_LABEL_SPLIT;
}


// Decide labels based on pair F1
// Use maxPrecDrop >= 1.0 to disable
template <typename TInt, typename TRegionMap, typename TImagePtr> int
genBoundaryClassificationLabelF1 (
    double& mergeF1, double& splitF1, TRegionMap const& rmap,
    typename TRegionMap::Key r0, typename TRegionMap::Key r1,
    typename TRegionMap::Key r2, TImagePtr const& truthImage,
    bool tweak, double maxPrecDrop)
{
  std::vector<typename TRegionMap::Region const*>
      pSplitRegions{&rmap.find(r0)->second, &rmap.find(r1)->second};
  std::vector<typename TRegionMap::Region const*>
      pMergeRegions{&rmap.find(r2)->second};
  return genBoundaryClassificationLabelF1<TInt>
      (mergeF1, splitF1, pSplitRegions, pMergeRegions, truthImage,
       tweak, maxPrecDrop);
}


// Decide labels based on traditional Rand index
template <typename TInt, typename TRegion, typename TImagePtr> int
genBoundaryClassificationLabelRI (
    double& mergeRI, double& splitRI,
    std::vector<TRegion const*> const& pSplitRegions,
    std::vector<TRegion const*> const& pMergeRegions,
    TImagePtr const& truthImage)
{
  TInt mergeTP, mergeTN, mergeFP, mergeFN,
      splitTP, splitTN, splitFP, splitFN;
  stats::pairStats(splitTP, splitTN, splitFP, splitFN, pSplitRegions,
                   truthImage, {BG_VAL});
  stats::pairStats(mergeTP, mergeTN, mergeFP, mergeFN, pMergeRegions,
                   truthImage, {BG_VAL});
  stats::randIndex(splitRI, splitTP, splitTN, splitFP, splitFN);
  stats::randIndex(mergeRI, mergeTP, mergeTN, mergeFP, mergeFN);
  return mergeRI > splitRI ? BC_LABEL_MERGE: BC_LABEL_SPLIT;
}


// Decide labels based on pair F1
// Use maxPrecDrop >= 1.0 to disable
template <typename TInt, typename TRegionMap, typename TImagePtr> int
genBoundaryClassificationLabelRI (
    double& mergeRI, double& splitRI, TRegionMap const& rmap,
    typename TRegionMap::Key r0, typename TRegionMap::Key r1,
    typename TRegionMap::Key r2, TImagePtr const& truthImage)
{
  std::vector<typename TRegionMap::Region const*>
      pSplitRegions{&rmap.find(r0)->second, &rmap.find(r1)->second};
  std::vector<typename TRegionMap::Region const*>
      pMergeRegions{&rmap.find(r2)->second};
  return genBoundaryClassificationLabelRI<TInt>(
      mergeRI, splitRI, pSplitRegions, pMergeRegions, truthImage);
}


};
};

#endif
