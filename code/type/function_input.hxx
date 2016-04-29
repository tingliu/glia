#ifndef _glia_type_function_input_hxx_
#define _glia_type_function_input_hxx_

#include "type/sampler.hxx"

namespace {

template <typename T, typename U> void
helperSubsample (std::vector<T>& samples, std::vector<U>& targets, int n)
{
  if (n > 0) {
    std::vector<int> indices;
    glia::crange(indices, 0, 1, samples.size());
    std::random_shuffle(indices.begin(), indices.end());
    glia::reorder(samples, indices);
    glia::reorder(targets, indices);
    samples.resize(n);
    targets.resize(n);
  } else if (n == 0) {
    samples.clear();
    targets.clear();
  }
}

};


namespace glia {
namespace opt {

class FunctionInput : public Object {
 public:
  typedef Object Super;
  typedef FunctionInput Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  ~FunctionInput () override {}

  virtual int size() const = 0;

  virtual int dim () const {
    perr("Unexpected: opt::FunctionInput::dim() not implemented...");
    return 0;
  }

  // Return false if using all samples
  virtual BatchSampler::BatchType prepareBatch () = 0;

  virtual bool isUsingBatchSampler () const = 0;

  virtual void turnOnUseAll () { _useAll = true; }

  virtual void turnOffUseAll () { _useAll = false; }

 protected:
  bool _useAll = false;
};


template <typename TSamp, typename TTar>
class FunctionInputSampleTarget : public FunctionInput {
 public:
  typedef FunctionInput Super;
  typedef FunctionInputSampleTarget<TSamp, TTar> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  typedef TSamp Sample;
  typedef TTar Target;

  FunctionInputSampleTarget () {
    _samples = std::make_shared<std::vector<std::shared_ptr<Sample>>>();
    _targets = std::make_shared<std::vector<std::shared_ptr<Target>>>();
  }

  ~FunctionInputSampleTarget () override {}

  void subsample (int n) { helperSubsample(*_samples, *_targets, n); }

  bool isUsingBatchSampler () const override { return bool(_sampler); }

  template <typename TBSampler, typename ...Args> void
  setBatchSampler (Args const&... args)
  { _sampler = std::make_shared<TBSampler>(args...); }

  void resetBatchSampler () {
    _sampler.reset();
    _batch.reset();
  }

  BatchSampler* getBatchSampler () { return _sampler.get(); }

  BatchSampler::BatchType prepareBatch () override {
    if (!_sampler) { return BatchSampler::BatchType::None; }
    if (!_batch) { _batch = std::make_shared<Self>(); }
    BatchSampler::BatchType ret = _sampler->prepareBatch();
    auto const& indices = _sampler->getBatch();
    _batch->samples().clear();
    _batch->targets().clear();
    _batch->samples().reserve(indices.size());
    _batch->targets().reserve(indices.size());
    for (int i : indices) {
      _batch->samples().push_back(_samples->at(i));
      _batch->targets().push_back(_targets->at(i));
    }
    return ret;
  }

  virtual std::vector<std::shared_ptr<Sample>>& allSamples ()
  { return *_samples; }

  virtual std::vector<std::shared_ptr<Sample>> const& allSamples () const
  { return *_samples; }

  virtual std::vector<std::shared_ptr<Target>>& allTargets ()
  { return *_targets; }

  virtual std::vector<std::shared_ptr<Target>> const& allTargets () const
  { return *_targets; }

  virtual std::vector<std::shared_ptr<Sample>>& samples ()
  { return _useAll || !_batch ? allSamples() : _batch->samples(); }

  virtual std::vector<std::shared_ptr<Sample>> const& samples () const
  { return _useAll || !_batch ? allSamples() : _batch->samples(); }

  virtual std::vector<std::shared_ptr<Target>>& targets ()
  { return _useAll || !_batch ? allTargets() : _batch->targets(); }

  virtual std::vector<std::shared_ptr<Target>> const& targets () const
  { return _useAll || !_batch ? allTargets() : _batch->targets(); }

  int size () const override
  { return _useAll ? allTargets().size() : targets().size(); }

  virtual int allSize () const { return allTargets().size(); }

 protected:
  std::shared_ptr<std::vector<std::shared_ptr<Sample>>> _samples;
  std::shared_ptr<std::vector<std::shared_ptr<Target>>> _targets;
  std::shared_ptr<BatchSampler> _sampler;
  std::shared_ptr<Self> _batch;
};

};
};

#endif
