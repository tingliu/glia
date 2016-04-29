#ifndef _glia_util_mp_hxx_
#define _glia_util_mp_hxx_

#include "util/container.hxx"

namespace glia {

inline int nthreads ()
{
  int ret;
#ifdef GLIA_MT
# pragma omp parallel
# pragma omp critical
  ret = omp_get_num_threads();
#else
  ret = 1;
#endif
  return ret;
}


// f(i): function to execute based on index
// maxThreads: 0 to use OMP_NUM_THREADS
template <typename Func> void
parfor (int start, uint end, bool shuffle, Func f, uint maxThreads)
{
  uint n = end - start;
#ifdef GLIA_MT
  if (maxThreads > 0) { omp_set_num_threads(maxThreads); }
  if (shuffle) {
    std::vector<int> indices;
    crange(indices, start, 1, end);
    std::random_shuffle(indices.begin(), indices.end());
#pragma omp parallel for
    for (int ii = 0; ii < n; ++ii) { f(indices[ii]); }
  }
  else {
#pragma omp parallel for
    for (int i = start; i < end; ++i) { f(i); }
  }
#else
  for (int i = start; i < end; ++i) { f(i); }
#endif
}


// maxThreads: 0 to use OMP_NUM_THREADS
template <typename TMap, typename Func> void
parfor (TMap& m, bool shuffle, Func f, uint maxThreads)
{
  uint n = m.size();
  std::vector<typename TMap::iterator> mits;
  mits.reserve(n);
  for (auto mit = m.begin(); mit != m.end(); ++mit)
  { mits.push_back(mit); }
#ifdef GLIA_MT
  if (maxThreads > 0) { omp_set_num_threads(maxThreads); }
  if (shuffle) {
    std::vector<int> indices;
    crange(indices, 0, 1, n);
    std::random_shuffle(indices.begin(), indices.end());
#pragma omp parallel for
    for (int ii = 0; ii < n; ++ii) {
      int i = indices[ii];
      f(mits[i], i);
    }
  }
  else {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) { f(mits[i], i); }
  }
#else
  for (int i = 0; i < n; ++i) { f(mits[i], i); }
#endif
}


// maxThreads: 0 to use OMP_NUM_THREADS
template <typename TMap, typename Func> void
parfor (TMap const& m, bool shuffle, Func f, uint maxThreads)
{
  uint n = m.size();
  std::vector<typename TMap::const_iterator> mits;
  mits.reserve(n);
  for (auto mit = m.begin(); mit != m.end(); ++mit)
  { mits.push_back(mit); }
#ifdef GLIA_MT
  if (maxThreads > 0) { omp_set_num_threads(maxThreads); }
  if (shuffle) {
    std::vector<int> indices;
    crange(indices, 0, 1, n);
    std::random_shuffle(indices.begin(), indices.end());
#pragma omp parallel for
    for (int ii = 0; ii < n; ++ii) {
      int i = indices[ii];
      f(mits[i], i);
    }
  }
  else {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) { f(mits[i], i); }
  }
#else
  for (int i = 0; i < n; ++i) { f(mits[i], i); }
#endif
}

};

#endif
