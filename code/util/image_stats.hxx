#ifndef _glia_util_image_stats_hxx_
#define _glia_util_image_stats_hxx_

#include "glia_image.hxx"
#include "util/stats.hxx"
#include "util/container.hxx"
#include "type/region.hxx"

namespace glia {
namespace stats {

template <typename TPoints, typename TImagePtr> void
histc (std::vector<uint>& hc, TImagePtr const& image,
       TPoints const& points, uint bin,
       std::pair<double, double> const& range)
{
  hc.resize(bin, 0.0);
  double interval = (range.second - range.first) / bin;
  std::vector<double> bounds(bin);
  bounds[0] = interval;
  for (auto i = 1; i < bin; ++i)
  { bounds[i] = bounds[i - 1] + interval; }
  points.traverse([&image, &range, &bounds, &hc, bin]
                  (typename TPoints::Point const& p) {
                    auto val = image->GetPixel(p);
                    if (val > range.first && val < range.second) {
                      for (auto i = 0; i < bin; ++i) {
                        if (val < bounds[i]) {
                          ++hc[i];
                          break;
                        }
                      }
                    }
                    else if (val <= range.first) { ++hc[0]; }
                    else { ++hc[bin - 1]; }
                  });
}


template <typename TPoints, typename TImagePtr> void
hist (std::vector<double>& h, std::vector<uint>& hc,
      TImagePtr const& image, TPoints const& points,
      uint bin, std::pair<double, double> const& range)
{
  histc(hc, image, points, bin, range);
  uint n = points.size();
  if (n == 0) {
    h.resize(bin, 0.0);
    return;
  }
  std::transform(hc.begin(), hc.end(), h.begin(),
                 [n](uint c) -> double { return c / (double)n; });
}


template <typename TPoints, typename TImagePtr> void
hist (std::vector<double>& h, TImagePtr const& image,
      TPoints const& points, uint bin,
      std::pair<double, double> const& range)
{
  std::vector<uint> hc;
  hist(h, hc, image, points, bin, range);
}


// Compute variation of information of points against truth image
// Assume each region has a distinctive label
// Ignore pixels on truth with values in exlcuded set
template <typename TRegion, typename TImagePtr> double
vi (std::vector<TRegion const*> const& pRegions,
    TImagePtr const& image,
    std::unordered_set<TImageVal<TImagePtr>> const& excluded)
{
  typedef TImageVal<TImagePtr> Key;
  uint nPoint = 0; // Total number of points
  uint nPointKey = pRegions.size();
  std::unordered_map<Key, std::vector<uint>> countsGivenImage;
  // Compute counts of points given image
  for (int pointKey = 0; pointKey < nPointKey; ++pointKey) {
    nPoint += pRegions[pointKey]->size();
    pRegions[pointKey]->traverse
        ([&image, &countsGivenImage, pointKey, nPointKey, &excluded]
         (typename TRegion::Point const& p){
          TImageVal<TImagePtr> imKey = image->GetPixel(p);
          if (excluded.count(imKey) == 0) {
            auto cit = citerator
                (countsGivenImage, imKey, std::vector<uint>(nPointKey, 0));
            ++cit->second[pointKey];
          }
        });
  }
  // Compute P(point)
  std::vector<double> logCountsPoint(nPointKey, 0.0);
  for (auto const& cp: countsGivenImage) {
    for (int pointKey = 0; pointKey < nPointKey; ++pointKey)
    { logCountsPoint[pointKey] += cp.second[pointKey]; }
  }
  for (auto& p: logCountsPoint) { p = std::log2(p); }
  // VI = H(point | image) + H(image | point)
  //    = sum(P(p, i) * (logP(p) + logP(i) - 2 * logP(p,i)))
  //    = sum(C(p, i) * (logC(p) + logC(i) - 2 * logC(p, i))) / N
  double ret = 0.0;
  for (auto const& cp: countsGivenImage) {
    double logCountImage =
        std::accumulate(cp.second.begin(), cp.second.end(), 0.0);
    if (logCountImage >= FEPS) {
      logCountImage = std::log2(logCountImage);
      for (int pointKey = 0; pointKey < nPointKey; ++pointKey) {
        double pairCount = cp.second[pointKey];
        if (pairCount >= FEPS) {
          ret += pairCount * (logCountImage + logCountsPoint[pointKey]
                              - 2.0 * std::log2(pairCount));
        }
      }
    }
  }
  return ret / nPoint;
}


// Conditional entropy H(image1 | image0)
template <typename TImagePtr, typename TMaskPtr> double
centropy (TImagePtr const& image0, TImagePtr const& image1,
          TMaskPtr const& mask,
          std::unordered_set<TImageVal<TImagePtr>> const& excluded0,
          std::unordered_set<TImageVal<TImagePtr>> const& excluded1)
{
  typedef TImageVal<TImagePtr> Key;
  std::unordered_map<Key, uint> ucounts;
  std::unordered_map<Key, std::unordered_map<Key, uint>> bcounts;
  uint n = 0;
  TImageCIIt<TImagePtr>
      iit0(image0, image0->GetRequestedRegion()),
      iit1(image1, image1->GetRequestedRegion());
  while (!iit0.IsAtEnd()) {
    if (mask.IsNull() ||
        mask->GetPixel(iit0.GetIndex()) != MASK_OUT_VAL) {
      Key r0 = iit0.Get(), r1 = iit1.Get();
      if (excluded0.count(r0) == 0 && excluded1.count(r1) == 0) {
        ++n;
        ++(citerator(ucounts, r0, 0)->second);
        ++(citerator(citerator(bcounts, r0)->second, r1, 0)->second);
      }
    }
    ++iit0;
    ++iit1;
  }
  double ret = 0.0;
  for (auto const& up: ucounts) {
    auto bit0 = bcounts.find(up.first);
    if (bit0 != bcounts.end()) {
      for (auto const& bp: bit0->second) {
        ret += (double)bp.second * std::log2(up.second / bp.second);
      }
    }
  }
  return ret / n;
}


template <typename TImagePtr, typename TMaskPtr> double
vi (TImagePtr const& image0, TImagePtr const& image1,
    TMaskPtr const& mask,
    std::unordered_set<TImageVal<TImagePtr>> const& excluded0,
    std::unordered_set<TImageVal<TImagePtr>> const& excluded1)
{
  return
      centropy(image0, image1, mask, excluded0, excluded1) +
      centropy(image1, image0, mask, excluded1, excluded0);
}


template <typename TInt, typename TRegion, typename TImagePtr> void
pairStats
(TInt& nTruePos, TInt& nTrueNeg, TInt& nFalsePos, TInt& nFalseNeg,
 std::vector<TRegion const*> const& pRegions, TImagePtr const& image,
 std::unordered_set<TImageVal<TImagePtr>> const& excluded)
{
  typedef TImageVal<TImagePtr> Key;
  std::unordered_map<std::pair<Key, Key>, TInt> cmap;
  int n = pRegions.size();
  for (int i = 0; i < n; ++i) {
    pRegions[i]->traverse
        ([&cmap, &image, &excluded, i](typename TRegion::Point const& p)
         {
           auto key = image->GetPixel(p);
           if (excluded.count(key) == 0)
           { ++citerator(cmap, std::make_pair(i, key), 0)->second; }
         });
  }
  pairStats(nTruePos, nTrueNeg, nFalsePos, nFalseNeg, cmap,
            {}, excluded);
}


template <typename TInt, typename TRKey, typename TRegion,
          typename TImagePtr> void
pairStats (
    TInt& nTruePos, TInt& nTrueNeg, TInt& nFalsePos, TInt& nFalseNeg,
    std::vector<std::pair<std::pair<TRKey, TRegion const*>, TImagePtr>>
    const& keyRegionImagePairs,
    std::unordered_set<TImageVal<TImagePtr>> const& excluded)
{
  std::unordered_map<std::pair<TRKey, TImageVal<TImagePtr>>, TInt> cmap;
  for (auto const& keyRegionImagePair : keyRegionImagePairs) {
    keyRegionImagePair.first.second->traverse(
        [&cmap, &keyRegionImagePair, &excluded](
            typename TRegion::Point const& p) {
          auto key = keyRegionImagePair.second->GetPixel(p);
          if (excluded.count(key) == 0) {
            ++citerator(
                cmap, std::make_pair(
                    keyRegionImagePair.first.first, key), 0)->second;
          }
        });
  }
  pairStats(
      nTruePos, nTrueNeg, nFalsePos, nFalseNeg, cmap, {}, excluded);
}


template <typename TInt, typename TFloat, typename TRegion,
          typename TImagePtr> void
pairF1 (TFloat& f, TFloat& prec, TFloat& rec,
        std::vector<TRegion const*> const& pRegions,
        TImagePtr const& image,
        std::unordered_set<TImageVal<TImagePtr>> const& excluded)
{
  TInt TP, TN, FP, FN;
  pairStats(TP, TN, FP, FN, pRegions, image, excluded);
  precision(prec, TP, FP);
  recall(rec, TP, FN);
  f1(f, prec, rec);
}


template <typename TInt, typename TRegion, typename TImagePtr> double
pairF1 (
    std::vector<TRegion const*> const& pRegions, TImagePtr const& image,
    std::unordered_set<TImageVal<TImagePtr>> const& excluded)
{
  double f, precision, recall;
  pairF1<TInt>(f, precision, recall, pRegions, image, excluded);
  return f;
}


template <typename TInt, typename TImagePtr, typename TMaskPtr> void
pairStats (TInt& nTruePos, TInt& nTrueNeg, TInt& nFalsePos,
           TInt& nFalseNeg, TImagePtr const& image0,
           TImagePtr const& image1, TMaskPtr const& mask,
           std::unordered_set<TImageVal<TImagePtr>> const& excluded0,
           std::unordered_set<TImageVal<TImagePtr>> const& excluded1)
{
  typedef TImageVal<TImagePtr> Key;
  std::unordered_map<std::pair<Key, Key>, TInt> cmap;
  TImageCIIt<TImagePtr>
      iit0(image0, image0->GetRequestedRegion()),
      iit1(image1, image1->GetRequestedRegion());
  while (!iit0.IsAtEnd()) {
    if (mask.IsNull() ||
        mask->GetPixel(iit0.GetIndex()) != MASK_OUT_VAL) {
      auto key0 = iit0.Get();
      auto key1 = iit1.Get();
      if (excluded0.count(key0) == 0 && excluded1.count(key1) == 0)
      { ++citerator(cmap, std::make_pair(key0, key1), 0)->second; }
    }
    ++iit0;
    ++iit1;
  }
  pairStats(nTruePos, nTrueNeg, nFalsePos, nFalseNeg, cmap,
            excluded0, excluded1);
}


template <typename TInt, typename TFloat, typename TImagePtr,
          typename TMaskPtr> void
pairF1 (TFloat& f, TFloat& prec, TFloat& rec, TImagePtr const& image0,
        TImagePtr const& image1, TMaskPtr const& mask,
        std::unordered_set<TImageVal<TImagePtr>> const& excluded0,
        std::unordered_set<TImageVal<TImagePtr>> const& excluded1)
{
  TInt TP, TN, FP, FN;
  pairStats(TP, TN, FP, FN, image0, image1, mask, excluded0,
            excluded1);
  precision(prec, TP, FP);
  recall(rec, TP, FN);
  f1(f, prec, rec);
}


template <typename TRegion, typename TRefImagePtr> void
getOverlap (
    std::unordered_map<TImageVal<TRefImagePtr>, int>& overlaps,
    TRegion const& region, TRefImagePtr const& refImage)
{
  region.traverse(
      [&overlaps, &refImage](typename TRegion::Point const& p) {
        auto key = refImage->GetPixel(p);
        auto oit = overlaps.find(key);
        if (oit == overlaps.end()) { overlaps[key] = 1; } else {
          ++(oit->second);
        }
      });
}


template <typename TImagePtr0, typename TImagePtr1,
          typename TMaskPtr> void
getOverlap (
    std::unordered_map<
    std::pair<TImageVal<TImagePtr0>, TImageVal<TImagePtr1>>, int>& cmap,
    TImagePtr0 const& image0, TMaskPtr const& mask0,
    std::unordered_set<TImageVal<TImagePtr0>> const& excluded0,
    TImagePtr1 const& image1, TMaskPtr const& mask1,
    std::unordered_set<TImageVal<TImagePtr1>> const& excluded1)
{
  typedef TImageVal<TImagePtr0> Key0;
  typedef TImageVal<TImagePtr1> Key1;
  TImageCIIt<TImagePtr0> iit0(image0, image0->GetRequestedRegion());
  TImageCIIt<TImagePtr1> iit1(image1, image1->GetRequestedRegion());
  while (!iit0.IsAtEnd() && !iit1.IsAtEnd()) {
    if ((mask0.IsNull() ||
         mask0->GetPixel(iit0.GetIndex()) != MASK_OUT_VAL) &&
        (mask1.IsNull() ||
         mask1->GetPixel(iit1.GetIndex()) != MASK_OUT_VAL)) {
      auto key0 = iit0.Get();
      auto key1 = iit1.Get();
      if (excluded0.count(key0) == 0 && excluded1.count(key1) == 0)
      { ++citerator(cmap, std::make_pair(key0, key1), 0)->second; }
    }
    ++iit0;
    ++iit1;
  }
}

};
};

#endif
