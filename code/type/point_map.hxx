#ifndef _glia_type_point_map_hxx_
#define _glia_type_point_map_hxx_

#include "type/hash.hxx"
#include "type/object.hxx"

namespace glia {

template <typename TKey, typename TPoint>
class TPointMap
    : public Object,
      public std::unordered_map<TKey, std::vector<TPoint>> {
public:
  typedef std::vector<TPoint> Points;
  typedef Object SuperObject;
  typedef std::unordered_map<TKey, Points> Super;
  typedef TPointMap<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef TKey Key;
  typedef TPoint Point;
  ~TPointMap () override {}
};


template <typename TKey, typename TPoint>
class TPointPairMap
    : public Object,
      public std::unordered_map<std::pair<TKey, TKey>,
                                std::vector<TPoint>> {
public:
  typedef std::vector<TPoint> Points;
  typedef Object SuperObject;
  typedef std::unordered_map<std::pair<TKey, TKey>, Points> Super;
  typedef TPointPairMap<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef TKey Key;
  typedef TPoint Point;

protected:
  std::unordered_map<TKey, std::vector<std::pair<TKey, Points*>>> m_umap;

public:
  ~TPointPairMap () override {}

  using Super::operator[];
  using Super::find;

  // Initialize m_umap; use if need m_umap
  virtual void prepare () {
    if (!m_umap.empty()) { return; }
    m_umap.clear();
    for (auto& pp: *this) {
      m_umap[pp.first.first].push_back
          (std::make_pair(pp.first.second, &pp.second));
    }
  }

  virtual std::unordered_map<TKey, std::vector<std::pair<TKey, Points*>>>
      const& unary () const { return m_umap; }

  virtual
      std::unordered_map<TKey, std::vector<std::pair<TKey, Points*>>>&
      unary () { return m_umap; }

  // Valid after initializing m_rbnmap
  virtual std::vector<std::pair<TKey, Points*>>& operator[] (TKey k)
  { return m_umap[k]; }

  // Valid after initialzing m_rbnmap
  virtual typename
      std::unordered_map<TKey, std::vector<std::pair<TKey, Points*>>>
      ::iterator
      find (TKey k) { return m_umap.find(k); }

  // Valid after initialzing m_rbnmap
  virtual typename
      std::unordered_map<TKey, std::vector<std::pair<TKey, Points*>>>
      ::const_iterator
      find (TKey k) const { return m_umap.find(k); }
};


template <typename TKey, typename TPoint>
class TPointPtrMap
    : public Object,
      public std::unordered_map<TKey, std::vector<TPoint>*> {
 public:
  typedef std::vector<TPoint> Points;
  typedef Object SuperObject;
  typedef std::unordered_map<TKey, Points*> Super;
  typedef TPointPtrMap<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef TKey Key;
  typedef TPoint Point;

  TPointPtrMap () {}

  TPointPtrMap (std::pair<TKey, Points*> const& pp) { merge(pp); }

  ~TPointPtrMap () override{}

  virtual uint size () const {
    uint ret = 0;
    for (auto const& pp: *this) { ret += pp.second->size(); }
    return ret;
  }

  virtual uint mapSize () const { return Super::size(); }

  virtual bool empty () const { return size() == 0; }

  template <typename Func> void traverse (Func f)
  { for (auto& pp: *this) { for (auto& p: *pp.second) { f(p); } } }

  template <typename Func> void traverse (Func f) const {
    for (auto const& pp: *this)
    { for (auto const& p: *pp.second) { f(p); } }
  }

  virtual void merge (std::pair<TKey, Points*> const& pp)
  { if (pp.second) { Super::insert(pp); } }

  virtual void merge (Self const& pmap)
  { for (auto const& pp: pmap) { merge(pp); } }
};


template <typename TKey, typename TPoint>
class TPointPtrPairMap
    : public Object,
      public std::unordered_map<std::pair<TKey, TKey>,
                                std::vector<TPoint>*> {
 public:
  typedef std::vector<TPoint> Points;
  typedef Object SuperObject;
  typedef std::unordered_map<std::pair<TKey, TKey>, Points*> Super;
  typedef TPointPtrPairMap<TKey, TPoint> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef TKey Key;
  typedef std::pair<TKey, TKey> KeyPair;
  typedef TPoint Point;

  TPointPtrPairMap () {}

  TPointPtrPairMap (std::pair<std::pair<TKey, TKey>, Points*> const& pp)
  { merge(pp); }

  ~TPointPtrPairMap () override {}

  virtual uint size () const {
    uint ret = 0;
    for (auto const& pp: *this) { ret += pp.second->size(); }
    return ret;
  }

  virtual uint mapSize () const { return Super::size(); }

  virtual bool empty () const { return size() == 0; }

  template <typename Func> void traverse (Func f)
  { for (auto& pp: *this) { for (auto& p: *pp.second) { f(p); } } }

  template <typename Func> void traverse (Func f) const {
    for (auto const& pp: *this)
    { for (auto const& p: *pp.second) { f(p); } }
  }

  virtual void
  merge (std::pair<std::pair<TKey, TKey>, Points*> const& pp)
  { if (pp.second) { Super::insert(pp); } }

  virtual void merge (Self const& pmap)
  { for (auto const& pp: pmap) { merge(pp); } }
};

};

#endif
