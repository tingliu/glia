#ifndef _glia_util_random_hxx_
#define _glia_util_random_hxx_

#include "glia_base.hxx"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

namespace glia {
namespace random {

template <typename T, typename TInputIterator> void
sampleWithoutReplacement (
    std::vector<T>& output, TInputIterator inputBegin,
    TInputIterator inputEnd, int n)
{
  std::random_shuffle(inputBegin, inputEnd);
  output.reserve(output.size() + n);
  for (int i = 0; i < n; ++i) { output.push_back(*(inputBegin + i)); }
}


template <typename T> void
sampleWithReplacement (
    std::vector<T>& output, std::vector<T> const& input, int n,
    const long seed)
{
  boost::mt19937 rng(seed);
  boost::random::uniform_int_distribution<> dice(0, input.size() - 1);
  output.reserve(n);
  for (int i = 0; i < n; ++i) { output.push_back(input[dice(rng)]); }
}


inline int sample (std::vector<double> const& weights, const long seed)
{
  boost::mt19937 rng(seed);
  boost::random::discrete_distribution<>
      dist(weights.begin(), weights.end());
  return dist(rng);
}


inline boost::variate_generator
<boost::mt19937, boost::normal_distribution<>>
    gaussian (double mean, double stddev, const long seed)
{
  boost::mt19937 rng(seed);
  boost::normal_distribution<> dist(mean, stddev);
  return
      boost::variate_generator
      <boost::mt19937, boost::normal_distribution<>>(rng, dist);
}

};
};

#endif
