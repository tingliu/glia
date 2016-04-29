#ifndef _glia_util_linalg_hxx_
#define _glia_util_linalg_hxx_

#include "glia_base.hxx"
#include "Eigen/Dense"

namespace glia {

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>
    createEmptyEigenVectorMap (std::vector<T>& mem, int d)
{
  mem.clear();
  mem.resize(d);
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(mem.data(), d);
}


template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
    createEmptyEigenMatrixMap (std::vector<T>& mem, int r, int c)
{
  mem.clear();
  mem.resize(r * c);
  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
      mem.data(), r, c);
}


template <typename T>
std::vector<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>>
createEmptyEigenVectorMaps (std::vector<T>& mem, int n, int d)
{
  std::vector<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>> ret;
  ret.reserve(n);
  mem.clear();
  mem.resize(n * d);
  for (int i = 0; i < n; ++i) {
    ret.push_back(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        mem.data() + i * d, d));
  }
  return ret;
}


template <typename T> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
combineToEigenMatrix (
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> const& input)
{
  int n = input.size();
  int d = input.front().size();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(d, n);
  for (int i = 0; i < n; ++i) { ret.col(i) = input[i]; }
  return ret;
}


template <typename T> void
convertToEigenMatrix (
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& output,
    std::vector<std::vector<T>> const& input,
    std::vector<int> const& indices)
{
  int d = input.front().size();
  int n = indices.size();
  output.resize(d, n);
  for (int i = 0; i < n; ++i) {
    output.col(i) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        input[indices[i]].data(), d);
  }
}


template <typename T> void
convertToEigenMatrix (
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& output,
    std::vector<std::vector<T>> const& input)
{
  int d = input.front().size();
  int n = input.size();
  output.resize(d, n);
  for (int i = 0; i < n; ++i) {
    output.col(i) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        input[i].data(), d);
  }
}


template <typename T> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
convertToEigenMatrix (std::vector<std::vector<T>> const& input)
{
  int d = input.front().size();
  int n = input.size();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> output(d, n);
  for (int i = 0; i < n; ++i) {
    output.col(i) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        input[i].data(), d);
  }
  return output;
}


template <typename T> void
convertToEigenMatrix (
    Eigen::Matrix<T, Eigen::Dynamic, 1>& output,
    std::vector<T> const& input)
{
  output = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
      input.data(), input.size());
}


template <typename T> void
convertFromEigenMatrix (
    std::vector<T>& output,
    Eigen::Matrix<T, Eigen::Dynamic, 1> const& input)
{
  output.resize(input.size());
  std::copy_n(input.data(), input.size(), output.data());
}


template <typename T> std::vector<T>
convertFromEigenMatrix (Eigen::Matrix<T, Eigen::Dynamic, 1> const& input)
{
  std::vector<T> ret;
  convertFromEigenMatrix(ret, input);
  return ret;
}

};

#endif
