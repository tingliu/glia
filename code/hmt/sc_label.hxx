#ifndef _glia_hmt_sc_label_hxx_
#define _glia_hmt_sc_label_hxx_

#include "util/image_stats.hxx"

namespace glia {
namespace hmt {

const int SC_LABEL_TRUE = 1;
const int SC_LABEL_FALSE = -1;

template <typename TInt, typename TRegion, typename TImagePtr> int
genSectionClassificationLabelF1 (
    double& trueF1, double& falseF1,
    std::vector<std::pair<TRegion const*, TImagePtr>> const&
    regionImagePairs)
{
  typedef TImageVal<TImagePtr> Key;
  int n = regionImagePairs.size();
  std::vector<std::pair<std::pair<Key, TRegion const*>, TImagePtr>>
      trueKeyRegionImagePairs, falseKeyRegionImagePairs;
  trueKeyRegionImagePairs.reserve(n);
  falseKeyRegionImagePairs.reserve(n);
  for (int i = 0; i < n; ++i) {
    trueKeyRegionImagePairs.push_back(
        std::make_pair(
            std::make_pair(0, regionImagePairs[i].first),
            regionImagePairs[i].second));
    falseKeyRegionImagePairs.push_back(
        std::make_pair(
            std::make_pair(i, regionImagePairs[i].first),
            regionImagePairs[i].second));
  }
  TInt trueTP = 0, trueTN = 0, trueFP = 0, trueFN = 0;
  TInt falseTP = 0, falseTN = 0, falseFP = 0, falseFN = 0;
  stats::pairStats(
      trueTP, trueTN, trueFP, trueFN, trueKeyRegionImagePairs, {BG_VAL});
  stats::pairStats(
      falseTP, falseTN, falseFP, falseFN, falseKeyRegionImagePairs,
      {BG_VAL});
  double truePrec, trueRec, falsePrec, falseRec;
  stats::precision(truePrec, trueTP, trueFP);
  stats::recall(trueRec, trueTP, trueFN);
  stats::f1(trueF1, truePrec, trueRec);
  stats::precision(falsePrec, falseTP, falseFP);
  stats::recall(falseRec, falseTP, falseFN);
  stats::f1(falseF1, falsePrec, falseRec);
  return trueF1 >= falseF1 ? SC_LABEL_TRUE : SC_LABEL_FALSE;
}


template <typename TInt, typename TRegion, typename TImagePtr> int
genSectionClassificationLabelF1 (
    double& trueF1, double& falseF1,
    TRegion const& r0, TImagePtr const& truthImage0,
    TRegion const& r1, TImagePtr const& truthImage1)
{
  std::vector<std::pair<TRegion const*, TImagePtr>> regionImagePairs;
  regionImagePairs.reserve(2);
  regionImagePairs.emplace_back(std::make_pair(&r0, truthImage0));
  regionImagePairs.emplace_back(std::make_pair(&r1, truthImage1));
  return genSectionClassificationLabelF1<TInt>(
      trueF1, falseF1, regionImagePairs);
}

};
};

#endif
