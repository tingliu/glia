#ifndef _glia_type_optimizer_hxx_
#define _glia_type_optimizer_hxx_

#include "type/object.hxx"

namespace glia {
namespace opt {

class OptimizerProto : public Object {
 public:
  typedef Object Super;
  typedef OptimizerProto Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  bool displayProgress;

  OptimizerProto (bool displayProgress)
      : displayProgress(displayProgress) {}

  ~OptimizerProto () override {}

  virtual void run () = 0;
};


template <typename TFunc>
class TOptimizer : public OptimizerProto {
 public:
  typedef OptimizerProto Super;
  typedef TOptimizer<TFunc> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TFunc Function;
  typedef typename Function::Input Input;

  std::shared_ptr<Function> pFunc;
  std::shared_ptr<Input> pInput;

  TOptimizer (
      std::shared_ptr<Function> pFunc, std::shared_ptr<Input> pInput,
      bool displayProgress)
      : Super(displayProgress), pFunc(pFunc), pInput(pInput) {}

  ~TOptimizer () override {}
};

};
};

#endif
