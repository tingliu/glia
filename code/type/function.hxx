#ifndef _glia_type_function_hxx_
#define _glia_type_function_hxx_

#include "type/object.hxx"

namespace glia {
namespace opt {

class FunctionProto : public Object {
 public:
  typedef Object Super;
  typedef FunctionProto Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  FunctionProto () {}

  ~FunctionProto () override {}

  // Dimensionality of parameters
  virtual int dim () const = 0;

  // Update parameters
  virtual void update (double const* w, int d) = 0;

  // Safe copy from x
  virtual void scopy (Self const& x) {}

  // Pointer to parameter data
  virtual double* data () = 0;

  // Pointer to parameter data;
  virtual double const* data () const = 0;
};


class ConstFunction : public FunctionProto {
 public:
  typedef FunctionProto Super;
  typedef ConstFunction Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  ~ConstFunction () override {}

  virtual double operator() () = 0;

  virtual double operator() (double* g) = 0;
};


template <typename TInput>
class TFunction : public FunctionProto {
 public:
  typedef FunctionProto Super;
  typedef TFunction<TInput> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TInput Input;

  ~TFunction () override {}

  // Compute function value
  virtual double operator() (Input const& x) = 0;

  // Compute function gradient in terms of parameters
  virtual double operator() (double* g, Input const& x) = 0;
};


template <typename T>
struct ThresholdModelDistributor{
 public:
  int dim0, dim1;
  T threshold;

  ThresholdModelDistributor (int dim0_, int dim1_, T const& threshold_)
      : dim0(dim0_), dim1(dim1_), threshold(threshold_) {}

  int operator() (std::vector<T> const& x) {
    if (x[dim1] < threshold) { return 0; }
    else if (x[dim0] < threshold) { return 1; }
    else { return 2; }
  }
};

};
};

#endif
