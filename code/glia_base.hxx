#ifndef _glia_base_hxx_
#define _glia_base_hxx_

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <memory>
#include <cstdint>
#include <cfloat>
#include <climits>
#include <typeinfo>
#include <initializer_list>
#include <functional>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <vector>
#include <array>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <deque>
#include <thread>
#include <chrono>
#ifdef GLIA_MT
#include <omp.h>
#endif

namespace glia {

typedef std::size_t uint;
typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;

typedef uint32 Label;
typedef float Real;
typedef double FVal; // Feature value type
const FVal FVAL_MIN = DBL_MIN;
const FVal FVAL_MAX = DBL_MAX;

#ifndef GLIA_3D
const int DIMENSION = 2;
#else
const int DIMENSION = 3;
#endif

const double PI = 3.14159265359;
const double DUMMY = -1.0;
const double FEPS = 2.22e-16;
const double FMIN = std::numeric_limits<double>::min();
const double FMAX = std::numeric_limits<double>::max();
const double LOG_FEPS = 0.0;
const uint FLT_PREC = 8;
const uint MAX_STR_LEN = 1024;

int DO_WEIRD_STUFF = 0;  // Globally reserved flag

inline void perr (std::string const& msg) {
  std::cerr << msg << std::endl;
  exit(EXIT_FAILURE);
}

inline bool isfeq (double lhs, double rhs)
{ return std::fabs(lhs - rhs) < FEPS; }

inline double minimax (double x, double mini, double maxi)
{ return std::min(maxi, std::max(mini, x)); }

inline double sdivide (double lhs, double rhs, double dummy)
{ return std::fabs(rhs) >= FEPS? lhs / rhs: dummy; }

inline double slog (double x, double dummy)
{ return x > 0.0? std::log(x): dummy; }

inline double ssqrt (double x, double dummy)
{ return x >= 0.0? std::sqrt(x): dummy; }

template <typename ...Args> inline bool
f_true (Args const&...) { return true; }

template <typename ...Args> inline void
f_null (Args const&...) {}

template <typename T>
struct is_pointer { static const bool value = false; };

template <typename T>
struct is_pointer<T*> { static const bool value = true; };

template <typename T, typename U = T> inline bool
inv_comp (T const& lhs, U const& rhs) { return lhs > rhs; }

template <typename TOut, typename TIn> inline TOut
cast (TIn const& x) { return (TOut)x; }

template <typename ...Args> inline std::string
strprintf (const char* pattern, Args const&... args) {
  char buf[MAX_STR_LEN];
  std::snprintf(buf, sizeof(buf), pattern, args...);
  return std::string(buf);
}

template <typename... Args> void
disp (std::string const& msg, Args const&... args)
{
  printf(msg.c_str(), args...);
  printf("\n");
}

};

#endif
