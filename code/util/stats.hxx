#ifndef _glia_util_stats_hxx_
#define _glia_util_stats_hxx_

#include "glia_base.hxx"

namespace glia {
namespace stats {

inline double& plusEqual (double& lhs, double rhs)
{
  if ((lhs == FMAX && rhs >= 0.0) || (lhs >= 0.0 && rhs == FMAX))
  { lhs = FMAX; }
  else if ((lhs == -FMAX && rhs <= 0.0) ||
           (lhs <= 0.0 && rhs == -FMAX)) { lhs = -FMAX; }
  else { lhs += rhs; }
  return lhs;
}


template <typename TContainer> typename TContainer::value_type
min (TContainer const& data)
{
  auto ret = data.front();
  for (auto const& x: data) { if (x < ret) { ret = x; } }
  return ret;
}


template <typename TContainer> typename TContainer::value_type
max (TContainer const& data)
{
  auto ret = data.front();
  for (auto const& x: data) { if (x > ret) { ret = x; } }
  return ret;
}


template <typename TContainer, typename T = double> T
sum (TContainer const& data)
{
  T ret = 0.0;
  for (auto const& x: data) { ret += x; }
  return ret;
}


template <typename TContainer> void
normalize (TContainer& data)
{
  double s = sum(data);
  if (isfeq(s, 0.0)) { return; }
  for (auto& x: data) { x /= s; }
}


template <typename TContainer> inline double
mean (TContainer const& data)
{ return sdivide(sum(data), data.size(), 0.0); }


template <typename TContainer> double
var (TContainer const& data, double mean)
{
  double ret = 0.0;
  for (auto const& x: data) {
    double dx = x - mean;
    ret += dx * dx;
  }
  return sdivide(ret, data.size(), 0.0);
}


template <typename TContainer> inline double
smedian (TContainer const& data)
{
  if (data.empty()) { return DUMMY; }
  auto it = data.begin();
  std::advance(it, data.size() / 2);
  return data.size() / 2 == 0? ((*it + *prev(it)) / 2.0): *it;
}


template <typename TContainer> inline typename TContainer::value_type
amedian (TContainer& data)
{
  if (data.empty()) { return DUMMY; }
  std::random_shuffle(data.begin(), data.end());
  std::nth_element(data.begin(), data.begin() + data.size() / 2,
                   data.end());
  return *(data.begin() + data.size() / 2);
}


template <typename TContainer> void
histc (std::vector<uint>& hc, TContainer const& data, uint bin,
       std::pair<double, double> const& range)
{
  hc.resize(bin, 0);
  if (data.empty()) { return; }
  double interval = (range.second - range.first) / bin;
  std::vector<double> bounds(bin);
  bounds[0] = interval;
  for (auto i = 1; i < bin; ++i)
  { bounds[i] = bounds[i - 1] + interval; }
  for (auto it = data.begin(); it != data.end(); ++it) {
    if (*it > range.first && *it < range.second) {
      for (auto i = 0; i < bin; ++i) {
        if (*it < bounds[i]) {
          ++hc[i];
          break;
        }
      }
    }
    else if (*it <= range.first) { ++hc[0]; }
    else { ++hc[bin - 1]; }
  }
}


template <typename TContainer> void
hist (std::vector<double>& h, std::vector<uint>& hc,
      TContainer const& data, uint bin,
      std::pair<double, double> const& range)
{
  histc(hc, data, bin, range);
  uint n = data.size();
  h.resize(bin, 0.0);
  if (n == 0) {
    return;
  }
  std::transform(hc.begin(), hc.end(), h.begin(),
                 [n](uint c) -> double { return c / (double)n;});
}


template <typename TContainer> void
hist (std::vector<double>& h, TContainer const& data, uint bin,
      std::pair<double, double> const& range)
{
  std::vector<uint> hc;
  hist(h, hc, data, bin, range);
}


template <typename TContainer> double
entropy (TContainer const& data)
{
  double ret = 0.0;
  for (auto const& p: data)
  { if (!isfeq(p, 0.0)) { ret -= p * log2(p);  } }
  return ret;
}


template <typename TContainer> double
distL1 (TContainer const& data0, TContainer const& data1)
{
  double ret = 0.0;
  auto it1 = data1.begin();
  for (auto it0 = data0.begin(); it0 != data0.end(); ++it0, ++it1)
  { ret += std::fabs(*it0 - *it1); }
  return ret;
}


template <typename TContainer> double
distL2 (TContainer const& data0, TContainer const& data1)
{
  double ret = 0.0;
  auto it1 = data1.begin();
  for (auto it0 = data0.begin(); it0 != data0.end(); ++it0, ++it1)
  { ret += std::pow(*it0 - *it1, 2); }
  return ret;
}


template <typename TContainer> double
distX2 (TContainer const& data0, TContainer const& data1)
{
  double ret = 0.0;
  auto it1 = data1.begin();
  for (auto it0 = data0.begin(); it0 != data0.end(); ++it0, ++it1)
  { ret += std::pow(*it0 - *it1, 2) / (*it0 + *it1 + FEPS); }
  return ret;
}


// Consider 0 as res and 1 as ref
template <typename TInt, typename TKey> void
pairStats
(TInt& nTruePos, TInt& nTrueNeg, TInt& nFalsePos, TInt& nFalseNeg,
 std::unordered_map<std::pair<TKey, TKey>, TInt> const& cmap,
 std::unordered_set<TKey> const& excluded0,
 std::unordered_set<TKey> const& excluded1)
{
  nTruePos = 0;
  TInt n = 0;
  std::unordered_set<TKey> keys0, keys1;
  for (auto const& cp: cmap) {
    n += cp.second;
    keys0.insert(cp.first.first);
    keys1.insert(cp.first.second);
    if (excluded0.count(cp.first.first) == 0 &&
        excluded1.count(cp.first.second) == 0)
    { nTruePos += cp.second * (cp.second - 1) / 2; }
  }
  TInt nPair = n * (n - 1) / 2,
      nPairWithIdenticalKey1 = 0, nPairWithIdenticalKey0 = 0;
  for (auto key1: keys1) {
    TInt _n = 0;
    for (auto key0: keys0) {
      auto cit = cmap.find(std::make_pair(key0, key1));
      if (cit != cmap.end()) { _n += cit->second; }
    }
    nPairWithIdenticalKey1 += _n * (_n - 1) / 2;
  }
  for (auto key0: keys0) {
    TInt _n = 0;
    for (auto key1: keys1) {
      auto cit = cmap.find(std::make_pair(key0, key1));
      if (cit != cmap.end()) { _n += cit->second; }
    }
    nPairWithIdenticalKey0 += _n * (_n - 1) / 2;
  }
  nTrueNeg = nPair - nPairWithIdenticalKey1 + nTruePos  -
      nPairWithIdenticalKey0;
  nFalsePos = nPairWithIdenticalKey0 - nTruePos;
  nFalseNeg = nPairWithIdenticalKey1 - nTruePos;
}


template <typename TFloat, typename TInt> void
randIndex (TFloat& ri, TInt const& nTruePos, TInt const& nTrueNeg,
           TInt const& nFalsePos, TInt const& nFalseNeg)
{
  TFloat num(nTruePos + nTrueNeg);
  TFloat den(nFalsePos + nFalseNeg);
  den += num;
  ri = num / (!isfeq(den, 0.0)? den: (den + FEPS));
}


template <typename TFloat, typename TInt> void
precision (TFloat& prec, TInt const& nTruePos, TInt const& nFalsePos)
{
  TFloat den(nTruePos + nFalsePos);
  prec = TFloat(nTruePos) / (!isfeq(den, 0.0)? den: (den + FEPS));
}


template <typename TFloat, typename TInt> void
recall (TFloat& rec, TInt const& nTruePos, TInt const& nFalseNeg)
{
  TFloat den(nTruePos + nFalseNeg);
  rec = TFloat(nTruePos) / (!isfeq(den, 0.0)? den: (den + FEPS));
}


template <typename TFloat> void
f1 (TFloat& f, TFloat const& prec, TFloat const& rec)
{ f =  2.0 * prec * rec / (prec + rec); }


template <typename TContainer> void
rescale (TContainer& feat, std::vector<std::vector<FVal>> const& minmax,
         FVal outputMin, FVal outputMax) {
  // Rescale
  FVal outputDiff = outputMax - outputMin;
  auto minit = minmax[0].begin();
  auto maxit = minmax[1].begin();
  for (auto& f : feat) {
    f = outputDiff * (f - *minit) / (*maxit - *minit + FEPS) +
        outputMin;
    ++minit;
    ++maxit;
  }
}


template <typename TContainer> void
rescale (
    std::vector<std::vector<FVal>>& minmax,
    std::vector<std::vector<TContainer>>& feats,
    FVal outputMin, FVal outputMax) {
  // No min/max provided, find them
  if (minmax.empty()) {
    int d = -1;
    for (auto const& ff0 : feats) {
      if (!ff0.empty()) {
        for (auto const& ff1 : ff0) {
          if (!ff1.empty()) {
            d = ff1.size();
            break;
          }
        }
        if (d > 0) { break; }
      }
    }
    minmax.resize(2);
    minmax[0].resize(d, FVAL_MAX);  // Min
    minmax[1].resize(d, FVAL_MIN);  // Max
    for (auto const& ff0 : feats) {
      for (auto const& ff1 : ff0) {
        auto minit = minmax[0].begin();
        auto maxit = minmax[1].begin();
        for (auto const& f : ff1) {
          if (f < *minit) { *minit = f; }
          if (f > *maxit) { *maxit = f; }
          ++minit;
          ++maxit;
        }
      }
    }
  }
  // Rescale
  for (auto& ffs : feats)
  { for (auto& ff : ffs) { rescale(ff, minmax, outputMin, outputMax); } }
}

};
};

#endif
