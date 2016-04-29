#ifndef _glia_type_boundary_table_hxx_
#define _glia_type_boundary_table_hxx_

#include "type/region_map.hxx"

namespace glia {

template <typename T, typename TRegionMap>
class TBoundaryTable : public Object {
 public:
  typedef Object SuperObject;
  typedef TBoundaryTable<T, TRegionMap> Self;
  typedef std::shared_ptr<Self> Pointer;
  typedef std::shared_ptr<const Self> ConstPointer;
  typedef std::weak_ptr<Self> WeakPointer;
  typedef typename TRegionMap::Key Key;

  struct Item;

  typedef std::map<std::pair<Key, Key>, std::shared_ptr<Item>> Table;
  typedef typename Table::iterator iterator;
  typedef typename Table::const_iterator const_iterator;
  typedef typename Table::value_type value_type;

  struct Item {
    typedef T Data;
    typename std::multimap<double, iterator>::iterator mqit;
    T data;
  };

 protected:
  Table m_table;
  std::multimap<double, iterator> m_mqueue;
  std::vector<double> m_weights;
  std::vector<iterator> m_items;

 public:
  TBoundaryTable () {}

  template <typename BFunc, typename SFunc>
  TBoundaryTable (TRegionMap const& rmap, BFunc fb, SFunc fsal)
  { init(rmap, fb, fsal); }

  ~TBoundaryTable () override {}

  // Return iterator to first table item with fcond(.) returning true
  template <typename CFunc> iterator
  top (CFunc fcond) {
    for (auto it = m_mqueue.rbegin(); it != m_mqueue.rend(); ++it)
    { if (fcond(*this, it->second)) { return it->second; } }
    return m_table.end();
  }

  // fsamp(std::vector<double>, const long seed): sampling function
  // fcond(*this, iterator): conditional pass function
  template <typename CFunc, typename SFunc> iterator
  top (CFunc fcond, SFunc fsamp, const long seed) {
    m_weights.clear();
    m_items.clear();
    m_weights.reserve(m_mqueue.size());
    m_items.reserve(m_mqueue.size());
    for (auto it = m_mqueue.rbegin(); it != m_mqueue.rend(); ++it) {
      m_weights.push_back(it->first);
      m_items.push_back(it->second);
    }
    while (!m_weights.empty()) {
      int i = fsamp(m_weights, seed);
      if (fcond(*this, m_items[i])) { return m_items[i]; }
      remove(m_weights, i);
      remove(m_items, i);
    }
    return m_table.end();
  }

  virtual Table const& table () const { return m_table; }

  virtual std::multimap<double, iterator> const& mqueue () const
  { return m_mqueue; }

  virtual std::multimap<double, iterator>& mqueue ()
  { return m_mqueue; }

  virtual uint size () const { return m_table.size(); }

  virtual bool empty () const { return m_table.empty(); }

  // fb: boundary table item data initializer
  // void fb (T&, Key r0, Key r1);
  // fsal: saliency function
  // double fsal (T const&, Key r0, Key r1);
  template <typename BFunc, typename SFunc> void
  init (TRegionMap const& rmap, BFunc fb, SFunc fsal) {
    for (auto const& rp: rmap) {
      for (auto const& bp: rp.second.boundary) {
        auto r0 = bp.first.first;
        auto r1 = bp.first.second;
        auto key = std::make_pair(r0, r1);
        if (key.first > key.second) { std::swap(key.first, key.second); }
        // Bugfix: boundaries have to be mutual
        if (m_table.count(key) == 0 &&
            rmap.find(r1)->second.boundary.count
            (std::make_pair(r1, r0)) > 0) {
          auto btit = m_table.emplace
              (key, std::shared_ptr<Item>(new Item)).first;
          fb(btit->second->data, r0, r1);
        }
      }
    }
    for (auto btit = m_table.begin(); btit != m_table.end(); ++btit) {
      btit->second->mqit = m_mqueue.insert
          (std::make_pair(fsal(btit->second->data, btit->first.first,
                               btit->first.second), btit));
    }
  }

  // fb: boundary table item data updater
  // void fb (T& data2s, Key r0, Key rs, Key r1, Key r2,
  //          T* pData0s, T* pData1s);
  // fsal: saliency function
  // double fsal (T const& data2s, Key rs, Key r2);
  template <typename BFunc, typename SFunc> void
  update (iterator btit01, Key r2, BFunc fb, SFunc fsal) {
    auto r0 = btit01->first.first;
    auto r1 = btit01->first.second;
    m_mqueue.erase(btit01->second->mqit);
    m_table.erase(btit01);
    auto btit = m_table.begin();
    while (btit != m_table.end() && btit->first.first <= r1) {
      Key rs;
      if (btit->first.first == r0 || btit->first.first == r1)
      { rs = btit->first.second; }
      else if (btit->first.second == r0 || btit->first.second == r1)
      { rs = btit->first.first; }
      else {
        ++btit;
        continue;
      }
      iterator btit0s, btit1s;
      if (btit->first.first == r0 || btit->first.second == r0) {
        btit0s = btit;
        btit1s = m_table.find
            (r1 < rs? std::make_pair(r1, rs): std::make_pair(rs, r1));
      }
      else {
        btit0s = m_table.find
            (r0 < rs? std::make_pair(r0, rs): std::make_pair(rs, r0));
        btit1s = btit;
      }
      auto btit2s = m_table.emplace
          (std::make_pair(rs, r2), std::shared_ptr<Item>(new Item)).first;
      fb(btit2s->second->data, r0, r1, rs, r2,
         btit0s == m_table.end()? nullptr: &btit0s->second->data,
         btit1s == m_table.end()? nullptr: &btit1s->second->data);
      double sal = fsal(btit2s->second->data, rs, r2);
      btit2s->second->mqit =
          m_mqueue.insert(std::make_pair(sal, btit2s));
      if (btit0s != m_table.end()) {
        m_mqueue.erase(btit0s->second->mqit);
        if (btit0s != btit) { m_table.erase(btit0s); }
      }
      if (btit1s != m_table.end()) {
        m_mqueue.erase(btit1s->second->mqit);
        if (btit1s != btit) { m_table.erase(btit1s); }
      }
      btit = m_table.erase(btit);
    }
  }
};

};

#endif
