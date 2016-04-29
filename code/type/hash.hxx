#ifndef _glia_type_hash_hxx_
#define _glia_type_hash_hxx_

#include "glia_base.hxx"

template <typename T> inline void
hash_combine (std::size_t& seed, T const& x)
{
  std::hash<T> hasher;
  seed ^= hasher(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {

template <typename T, typename U>
struct hash<pair<T, U>> {
  size_t operator() (pair<T, U> const& pa) const {
    size_t seed = 0;
    ::hash_combine(seed, pa.first);
    ::hash_combine(seed, pa.second);
    return seed;
  }
};


template <typename T>
struct hash<vector<T>> {
  size_t operator() (vector<T> const& c) const {
    size_t seed = 0;
    for (auto const& x: c) { ::hash_combine(seed, x); }
    return seed;
  }
};

};

#endif
