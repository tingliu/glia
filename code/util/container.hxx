#ifndef _glia_util_container_hxx_
#define _glia_util_container_hxx_

#include "glia_base.hxx"

namespace glia {

template <typename T, typename Func> void
inplace_op (T* p, int n, Func f)
{
  T* pEnd = p + n;
  while (p != pEnd) {
    f(*p);
    ++p;
  }
}

template <typename TOut, typename TIn, typename Func> void
unary_op (TOut* pOut, TIn const* pIn, int n, Func f)
{
  TOut* pOutEnd = pOut + n;
  while (pOut != pOutEnd) {
    f(*pOut, *pIn);
    ++pOut;
    ++pIn;
  }
}

template <typename TIn0, typename TIn1, typename Func> void
binary_op (TIn0 const* pIn0, TIn1 const* pIn1, int n, Func f)
{
  TIn0 const* pIn0End = pIn0 + n;
  while (pIn0 != pIn0End) {
    f(*pIn0, *pIn1);
    ++pIn0;
    ++pIn1;
  }
}

template <typename TOut, typename TIn0, typename TIn1, typename Func> void
binary_op (TOut* pOut, TIn0 const* pIn0, TIn1 const* pIn1, int n, Func f)
{
  TOut* pOutEnd = pOut + n;
  while (pOut != pOutEnd) {
    f(*pOut, *pIn0, *pIn1);
    ++pOut;
    ++pIn0;
    ++pIn1;
  }
}


template <typename T> T
opSquaredNorm (T* p, int n)
{
  T ret = 0.0;
  inplace_op(p, n, [&ret](T const& x) { ret += x * x; });
  return ret;
}


template <typename TContainer> inline typename TContainer::mapped_type*
cpointer (TContainer& c, typename TContainer::key_type const& key)
{
  auto it = c.find(key);
  return it == c.end()? nullptr: &it->second;
}


template <typename TContainer> inline
typename TContainer::mapped_type const*
ccpointer (TContainer const& c, typename TContainer::key_type const& key)
{
  auto it = c.find(key);
  return it == c.end()? nullptr: &it->second;
}


template <typename TContainer, typename... Args>
inline typename TContainer::iterator
citerator (TContainer& c, typename TContainer::key_type const& key,
           Args... args)
{
  auto it = c.find(key);
  if (it == c.end()) {
    it = c.insert(c.end(), std::make_pair
                  (key, typename TContainer::mapped_type(args...)));
  }
  return it;
}


template <typename TContainer> void
cciterators (std::vector<typename TContainer::const_iterator>& its,
             TContainer const& c)
{
  its.reserve(its.size() + c.size());
  for (auto it = c.begin(); it != c.end(); ++it) { its.push_back(it); }
}


template <typename TContainer>
inline typename TContainer::mapped_type
clookup (TContainer const& c, typename TContainer::key_type const& key,
         typename TContainer::mapped_type const& defaultVal)
{
  auto it = c.find(key);
  return it == c.end()? defaultVal: it->second;
}


// Generate range [start: step: end)
template <typename T, typename U> void
crange (std::vector<T>& c, T start, T step, U end)
{
  c.clear();
  c.reserve(std::floor(((double)end - (double)start) / step));
  T x = start;
  while (x < end - FEPS) {
    c.push_back(x);
    x += step;
  }
}


template <typename TMap> void
ckeys (std::vector<typename TMap::key_type>& keys, TMap const& c)
{
  keys.reserve(keys.size() + c.size());
  for (auto const& p: c) { keys.push_back(p.first); }
}


template <typename TMap> typename TMap::key_type
cmaxkey (TMap const& c)
{
  auto ret = c.begin()->first;
  for (auto const& p: c) { ret = std::max(ret, p.first); }
  return ret;
}


// Insert empty elements and return pointers
template <typename TMap> void
cinsert (std::vector<typename TMap::mapped_type*>& ps, TMap& c,
         std::vector<typename TMap::key_type> const& keys)
{
  ps.reserve(ps.size() + keys.size());
  for (auto const& key: keys) { ps.push_back(&c[key]); }
}


// f: returns number of elements in each data
template <typename TContainer, typename Func> uint
count (TContainer const& c, Func f)
{
  uint ret = 0;
  for (auto const& x: c) { ret += f(c); }
  return ret;
}


template <typename T> void
clinearize (std::vector<T*>& pcs,
            std::vector<std::vector<std::vector<T>>>& c)
{
  uint n = 0;
  for (auto const& cc: c)
  { for (auto const& ccc: cc) { n += ccc.size(); } }
  pcs.reserve(pcs.size() + n);
  for (auto& cc: c) {
    for (auto& ccc: cc)
    { for (auto& x: ccc) { pcs.push_back(&x); } }
  }
}


template <typename T> void
clinearize (std::vector<T const*>& pcs,
            std::vector<std::vector<std::vector<T>>> const& c)
{
  uint n = 0;
  for (auto const& cc: c)
  { for (auto const& ccc: cc) { n += ccc.size(); } }
  pcs.reserve(pcs.size() + n);
  for (auto const& cc: c) {
    for (auto const& ccc: cc)
    { for (auto const& x: ccc) { pcs.push_back(&x); } }
  }
}


// Fast remove element from vector without maintaining order
template <typename T> inline void
remove (std::vector<T>& c, int i)
{
  std::swap(c[i], c.back());
  c.resize(c.size() - 1);
}


template <typename T> void
remove (std::vector<T>& c, std::vector<int>& indices)
{
  std::sort(indices.begin(), indices.end());
  for (auto iit = indices.rbegin(); iit != indices.rend(); ++iit)
  { remove(c, *iit); }
}


// Indices must be in ascending order
// So that removal can be done in reverse order
template <typename T> void
remove (std::vector<T>& c, std::set<int>& indices)
{
  for (auto iit = indices.rbegin(); iit != indices.rend(); ++iit)
  { remove(c, *iit); }
}


template <typename T> inline void
splice (std::vector<T>& dst, std::vector<T>& src)
{
  if (src.empty()) { return; }
  if (dst.empty()) { dst = std::move(src); }
  else {
    dst.reserve(dst.size() + src.size());
    std::move(src.begin(), src.end(), std::back_inserter(dst));
  }
  src.clear();
}


template <typename T> inline void
splice (std::vector<T>& dst, std::vector<T>& src0, std::vector<T>& src1)
{
  if (src0.empty()) { splice(dst, src1); }
  else if (src1.empty()) { splice(dst, src0); }
  else {
    dst.reserve(dst.size() + src0.size() + src1.size());
    std::move(src0.begin(), src0.end(), std::back_inserter(dst));
    std::move(src1.begin(), src1.end(), std::back_inserter(dst));
    src0.clear();
    src1.clear();
  }
}


template <typename T> void
concat (std::vector<T>& dst, std::vector<std::vector<T>>& srcs)
{
  uint n = 0;
  for (auto const& c: srcs) { n += c.size(); }
  dst.reserve(n);
  for (auto& c: srcs) { for (auto& x: c) { dst.push_back(x); } }
  srcs.clear();
}


template <typename T> inline void
append (std::vector<T>& dst, std::vector<T> const& src)
{
  dst.reserve(dst.size() + src.size());
  std::copy(src.begin(), src.end(), std::back_inserter(dst));
}


template <typename T> inline void
append (std::vector<T>& dst, std::vector<T> const& src0,
        std::vector<T> const& src1)
{
  dst.reserve(dst.size() + src0.size() + src1.size());
  std::copy(src0.begin(), src0.end(), std::back_inserter(dst));
  std::copy(src1.begin(), src1.end(), std::back_inserter(dst));
  // for (auto const& x: src0) { dst.push_back(x); }
  // for (auto const& x: src1) { dst.push_back(x); }
}


template <typename T> void
add (std::vector<T>& res, std::vector<T> const& c0,
     std::vector<T> const& c1)
{
  assert("Error: container sizes disagree..." && c0.size() == c1.size());
  res.resize(c0.size());
  for (auto it = res.begin(), it0 = c0.begin(), it1 = c1.begin();
       it0 != c0.end(); ++it, ++it0, ++it1) { *it = *it0 + *it1; }
}


template <typename TContainer> bool
compare (TContainer const& c0, TContainer const& c1)
{
  auto cit0 = c0.begin();
  auto cit1 = c1.begin();
  while (cit0 != c0.end() && cit1 != c1.end()) {
    if (*cit0 < *cit1) { return true; }
    else if (*cit0 > *cit1) { return false; }
    ++cit0;
    ++cit1;
  }
  if (cit0 == c0.end() && cit1 != c1.end()) { return true; }
  return false;
}


template <typename T, typename TIndexContainer> void
select (std::vector<T>& output, std::vector<T> const& input,
        TIndexContainer const& indices)
{
  output.reserve(output.size() + indices.size());
  for (int i : indices) { output.push_back(input[i]); }
}


template <typename T> void
reorder (std::vector<T>& c, std::vector<int> const& indices_)
{
  std::vector<int> indices = indices_;
  int n = indices_.size();
  T x;
  for (int i = 0; i < n; ++i) {
    x = std::move(c[i]);
    int j = i;
    while (true) {
      int k = indices[j];
      indices[j] = j;
      if (k == i) { break; }
      c[j] = std::move(c[k]);
      j = k;
    }
    c[j] = std::move(x);
  }
}

// Monotonically increase x
// If n < x.size(), do nothing
// If keep, previous elements are not destroyed
template <typename T> void
incvec (std::vector<T>& x, int n, bool keep)
{
  if (x.size() < n) {
    if (keep) {
      std::vector<T> x_(n);
      int nn = x.size();
      for (int i = 0; i < nn; ++i) { x_[i] = std::move(x[i]); }
      x.swap(x_);
    } else {
      x.clear();
      x.resize(n);
    }
  }
}

template <typename T> T
majority (std::vector<T> const& x)
{
  if (x.empty()) {
    perr("Error: empty pool for selecting majority...");
  }
  std::unordered_map<T, int> counts;
  for (auto const& xx : x) {
    auto it = counts.find(xx);
    if (it == counts.end()) {
      counts[xx] = 1;
    } else { ++it->second; }
  }
  int maxCount = -1;
  T ret;
  for (auto const& c : counts) {
    if (c.second > maxCount) {
      maxCount = c.second;
      ret = c.first;
    }
  }
  return ret;
}

};

#endif
