#ifndef _glia_type_sampler_hxx_
#define _glia_type_sampler_hxx_

#include "type/object.hxx"
#include "util/container.hxx"

namespace glia {

class BatchSampler : public Object {
 public:
  typedef Object Super;
  typedef BatchSampler Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  enum class BatchType : int {
    None = 0,
    NewBatch = 1,
    EndOfEpoch = 2,
  };

  virtual ~BatchSampler () = 0;

  virtual int getBatchSize () const { return _batchSize; }

  virtual std::vector<int> const& getBatch () { return _batch; }

  virtual void reset () = 0;

  virtual void initialize (int batchSize) {
    _batchSize = batchSize;
    _batch.reserve(_batchSize);
  }

  virtual BatchType prepareBatch () = 0;

 protected:
  int _batchSize = -1;
  std::vector<int> _batch;
};

inline BatchSampler::~BatchSampler () {}


class UniformBatchSampler : public BatchSampler {
 public:
  typedef BatchSampler Super;
  typedef UniformBatchSampler Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  void reset () override {
    std::random_shuffle(_all.begin(), _all.end());
    _curIndex = 0;
    _allVisited = false;
  }

  virtual void initialize (int totalSize, int batchSize) {
    if (totalSize < batchSize)
    { perr("Error: totalSize must be greater than batchSize"); }
    Super::initialize(batchSize);
    _totalSize = totalSize;
    crange(_all, 0, 1, _totalSize);
    reset();
  }

  UniformBatchSampler () {}

  UniformBatchSampler (int totalSize, int batchSize)
  { initialize(totalSize, batchSize); }

  ~UniformBatchSampler () override {}

  virtual BatchType prepareBatch (int n) {
    if (n <= 0) { return BatchType::None; }
    BatchType ret = BatchType::NewBatch;
    _batch.clear();
    for (int i = 0; i < n; ++i) {
      _batch.push_back(_all[_curIndex++]);
      if (_curIndex >= _totalSize) {
        _allVisited = true;
        _curIndex = 0;
      }
    }
    if (_allVisited) {
      reset();
      ret = BatchType::EndOfEpoch;
    }
    return ret;
  }

  BatchType prepareBatch () override { return prepareBatch(_batchSize); }

  virtual int getBatchNumPerEpoch () const
  { return (int)(std::ceil((double)_totalSize / _batchSize)); }

 protected:
  int _totalSize = -1;
  std::vector<int> _all;
  int _curIndex = -1;
  bool _allVisited = false;
};


class ClassBatchSampler : public BatchSampler {
 public:
  typedef BatchSampler Super;
  typedef ClassBatchSampler Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  void reset () override {
    std::fill(_classCurIndex.begin(), _classCurIndex.end(), 0);
    // If a class has batch size 0, always mark it as visited
    int cn = _classBatchSize.size();
    for (int i = 0; i < cn; ++i)
    { _classAllVisited[i] = (_classBatchSize[i] <= 0); }
    // Shuffle all indices
    for (auto& indices : _classIndices)
    { std::random_shuffle(indices.begin(), indices.end()); }
  }

  virtual void initialize (
      std::vector<int>& classBatchSize,
      std::vector<std::vector<int>>& classIndices) {
    _classBatchSize.swap(classBatchSize);
    _classIndices.swap(classIndices);
    _classCurIndex.resize(_classBatchSize.size());
    _classAllVisited.resize(_classBatchSize.size());
    int batchSize = 0;
    for (int siz : _classBatchSize) { batchSize += siz; }
    Super::initialize(batchSize);
    reset();
  }

  template <typename TTar> void
  initialize (
      std::vector<std::pair<TTar, int>> const& targetBatchSize,
      std::vector<std::shared_ptr<TTar>> const& targets) {
    int cn = targetBatchSize.size();
    std::unordered_map<TTar, int> targetIndex;
    std::vector<int> classBatchSize;
    classBatchSize.reserve(cn);
    for (auto const& ts : targetBatchSize) {
      targetIndex[ts.first] = classBatchSize.size();
      classBatchSize.push_back(ts.second);
    }
    int tn = targets.size();
    std::vector<std::vector<int>> classIndices(cn);
    for (int i = 0; i < tn; ++i) {
      int ti = targetIndex.find(*targets[i])->second;
      classIndices[ti].push_back(i);
    }
    initialize(classBatchSize, classIndices);
  }

  // Compute batch sizes as: discount * min(#class)
  template <typename TTar> void
  initializeBalanceBatch (
      std::vector<std::shared_ptr<TTar>> const& targets,
      double discount) {
    std::map<TTar, int> targetCounts;
    for (auto const& tar : targets) {
      auto it = targetCounts.find(*tar);
      if (it == targetCounts.end()) { targetCounts[*tar] = 1; }
      else { ++(it->second);}
    }
    int batchSize = INT_MAX;
    for (auto const& tc : targetCounts)
    { batchSize = std::min(batchSize, tc.second); }
    batchSize *= discount;
    std::vector<std::pair<TTar, int>> targetBatchSize;
    targetBatchSize.reserve(targetCounts.size());
    for (auto const& tc : targetCounts)
    { targetBatchSize.push_back(std::make_pair(tc.first, batchSize)); }
    initialize(targetBatchSize, targets);
  }

  ClassBatchSampler () {}

  template <typename TTar>
  ClassBatchSampler (
      std::vector<std::pair<TTar, int>> const& targetBatchSize,
      std::vector<std::shared_ptr<TTar>> const& targets)
  { initialize(targetBatchSize, targets); }

  // Special constructor: initialize to balanced batch
  template <typename TTar>
  ClassBatchSampler (
      std::vector<std::shared_ptr<TTar>> const& targets, double discount)
  { initializeBalanceBatch(targets, discount); }

  ~ClassBatchSampler () {}

  virtual BatchType prepareBatch (
      std::vector<int> const& classBatchSize) {
    BatchType ret = BatchType::NewBatch;
    _batch.clear();
    int cn = classBatchSize.size();
    for (int i = 0; i < cn; ++i) {
      int& curIndex = _classCurIndex[i];
      bool doShuffle = false;
      for (int j = 0; j < classBatchSize[i]; ++j) {
        _batch.push_back(_classIndices[i][curIndex++]);
        if (curIndex >= _classIndices[i].size()) {
          _classAllVisited[i] = true;
          doShuffle = true;
          curIndex = 0;
        }
      }
      if (doShuffle) {
        // Shuffle each class indivdually
        std::random_shuffle(
            _classIndices[i].begin(), _classIndices[i].end());
      }
    }
    if (std::all_of(_classAllVisited.begin(), _classAllVisited.end(),
                    [](bool x) { return x; })) {
      reset();
      ret = BatchType::EndOfEpoch;
    }
    return ret;
  }

  BatchType prepareBatch () override
  { return prepareBatch(_classBatchSize); }

 protected:
  std::vector<int> _classBatchSize;
  std::vector<std::vector<int>> _classIndices;
  std::vector<int> _classCurIndex;
  std::vector<bool> _classAllVisited;
};

};

#endif
