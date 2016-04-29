#ifndef _glia_hmt_tree_segment_hxx_
#define _glia_hmt_tree_segment_hxx_

#include "util/image.hxx"
#include "util/container.hxx"

namespace glia {
namespace hmt {

template <typename TKey, typename TTr, typename TContainer> void
genLabelTransform (std::unordered_map<TKey, TKey>& lmap,
                   TTr const& tree, TContainer const& picks,
                   TKey keyToAssign)
{
  for (auto pick: picks) {
    tree.traverseLeaves(
        pick, [keyToAssign, &lmap](typename TTr::Node const& node) {
          lmap[node.data.label] = keyToAssign; });
    ++keyToAssign;
  }
}


template <typename TKey, typename TTr, typename TContainer> void
genLabelTransform (std::unordered_map<TKey, TKey>& lmap,
                   std::vector<TTr> const& trees,
                   TContainer const& picks, TKey keyToAssign)
{
  for (auto const& pick: picks) {
    trees[pick.first].traverseLeaves(
        pick.second, [keyToAssign, &lmap](typename TTr::Node const& node)
        { lmap[node.data.label] = keyToAssign; });
    ++keyToAssign;
  }
}


template <typename TImagePtr, typename TTr, typename TMaskPtr,
          typename TContainer> void
genFinalSegmentation
(TImagePtr& segImage, TTr const& tree, TContainer const& picks,
 TMaskPtr const& mask, TImageVal<TImagePtr> const& keyToAssign,
 bool exact)
{
  typedef TImageVal<TImagePtr> Key;
  std::unordered_map<Key, Key> lmap;
  genLabelTransform(lmap, tree, picks, keyToAssign);
  if (exact) { transformImage(segImage, lmap, mask); }
  else { transformImage(segImage, lmap, mask, true); }
}


template <typename TImagePtr, typename TTr, typename TMaskPtr,
          typename TContainer> void
genFinalSegmentation
(TImagePtr& segImage, std::vector<TTr> const& trees,
 TContainer const& picks, TMaskPtr const& mask,
 TImageVal<TImagePtr> const& keyToAssign, bool exact)
{
  typedef TImageVal<TImagePtr> Key;
  std::unordered_map<Key, Key> lmap;
  genLabelTransform(lmap, trees, picks, keyToAssign);
  if (exact) { transformImage(segImage, lmap, mask); }
  else { transformImage(segImage, lmap, mask, true); }
}


// Use all nodes' potentials if picks.empty() == true
template <typename TVal, typename TRegionMap, typename TTr,
          typename TContainer, typename Func> void
genBoundaryConfidenceMap
(std::unordered_map<std::pair<typename TRegionMap::Key,
 typename TRegionMap::Key>, TVal>& pbmap, TRegionMap const& rmap,
 TTr const& tree, TContainer const& picks, Func f)
{
  typedef typename TRegionMap::Key Key;
  auto fpb = [&pbmap, &f](double val, std::pair<Key, Key> const& key01)
      {
        auto key10 = key01.first < key01.second? key01:
        std::make_pair(key01.second, key01.first);
        auto pbit = pbmap.find(key10);
        if (pbit == pbmap.end()) { pbmap[key10] = val; }
        else if (pbit->second < val) { pbit->second = val; }
      };
  if (picks.empty()) { // Use all nodes
    for (auto const& tn: tree) {
      auto val = f(tn);
      for (auto const& bp:
               rmap.find(tn.data.label)->second.boundary)
      { fpb(val, bp.first); }
    }
  }
  else { // Only use picked nodes
    for (auto pick: picks) {
      auto const& tn = tree[pick];
      auto val = f(tn);
      for (auto const& bp:
               rmap.find(tn.data.label)->second.boundary)
      { fpb(val, bp.first); }
    }
  }
}


// Use all nodes' potentials if picks.empty() == true
template <typename TVal, typename TRegionMap, typename TTr,
          typename TContainer, typename Func> void
genBoundaryConfidenceMap
(std::unordered_map<std::pair<typename TRegionMap::Key,
 typename TRegionMap::Key>, TVal>& pbmap,
 std::vector<TRegionMap> const& rmaps,
 std::vector<TTr> const& trees, TContainer const& picks, Func f)
{
  typedef typename TRegionMap::Key Key;
  auto fpb = [&pbmap, &f](double val, std::pair<Key, Key> const& key01)
      {
        auto key10 = key01.first < key01.second? key01:
        std::make_pair(key01.second, key01.first);
        auto pbit = pbmap.find(key10);
        if (pbit == pbmap.end()) { pbmap[key10] = val; }
        else if (pbit->second < val) { pbit->second = val; }
      };
  int nTree = trees.size();
  if (picks.empty()) { // Use all nodes
    for (int i = 0; i < nTree; ++i) {
      for (auto const& tn: trees[i]) {
        auto val = f(tn);
        for (auto const& bp:
                 rmaps[i].find(tn.data.label)->second.boundary)
        { fpb(val, bp.first); }
      }
    }
  }
  else { // Only use picked nodes
    for (auto const& pick: picks) {
      auto const& tn = trees[pick.first][pick.second];
      auto val = f(tn);
      for (auto const& bp:
               rmaps[pick.first].find(tn.data.label)->second.boundary)
      { fpb(val, bp.first); }
    }
  }
}


template <typename TImagePtr, typename TRegionMap> void
genBoundaryConfidenceImage
(TImagePtr& bcImage, TRegionMap const& rmap,
 std::unordered_map<std::pair<typename TRegionMap::Key,
 typename TRegionMap::Key>, TImageVal<TImagePtr>> const& pbmap)
{
  for (auto const& pbp: pbmap) {
    auto key = pbp.first;
    auto rit = rmap.find(pbp.first.first);
    auto bit = rit->second.boundary.find(key);
    if (bit != rit->second.boundary.end()) {
      for (auto const& p: *bit->second) {
        if (bcImage->GetPixel(p) < pbp.second)
        { bcImage->SetPixel(p, pbp.second); }
      }
    }
    std::swap(key.first, key.second);
    rit = rmap.find(pbp.first.second);
    bit = rit->second.boundary.find(key);
    if (bit != rit->second.boundary.end()) {
      for (auto const& p: *bit->second) {
        if (bcImage->GetPixel(p) < pbp.second)
        { bcImage->SetPixel(p, pbp.second); }
      }
    }
  }
}


// Use all nodes' potentials if picks.empty() == true
template <typename TImagePtr, typename TRegionMap, typename TTr,
          typename TContainer, typename Func> void
genBoundaryConfidenceImage
(TImagePtr& bcImage, TRegionMap const& rmap, TTr const& tree,
 TContainer const& picks, Func f)
{
  typedef typename TRegionMap::Key Key;
  std::unordered_map<std::pair<Key, Key>, TImageVal<TImagePtr>> pbmap;
  genBoundaryConfidenceMap(pbmap, rmap, tree, picks, f);
  genBoundaryConfidenceImage(bcImage, rmap, pbmap);
}


// Use all nodes' potentials if picks.empty() == true
template <typename TImagePtr, typename TRegionMap, typename TTr,
          typename TContainer, typename Func> void
genBoundaryConfidenceImage
(TImagePtr& bcImage, std::vector<TRegionMap> const& rmaps,
 std::vector<TTr> const& trees, TContainer const& picks, Func f)
{
  typedef typename TRegionMap::Key Key;
  std::unordered_map<std::pair<Key, Key>, TImageVal<TImagePtr>> pbmap;
  genBoundaryConfidenceMap(pbmap, rmaps, trees, picks, f);
  genBoundaryConfidenceImage(bcImage, rmaps.front(), pbmap);
}


template <typename TKey, typename TVal, typename TThreshold> void
thresholdBoundaryConfidenceMap
(std::vector<TTriple<TKey>>& order,
 std::unordered_map<std::pair<TKey, TKey>, TVal> const& pbmap,
 TThreshold t, TKey keyToAssign)
{
  order.reserve(pbmap.size());
  std::unordered_map<TKey, TKey> lmap;
  for (auto const& pbp: pbmap) {
    if (pbp.second < t) {
      auto kit0 = citerator(lmap, pbp.first.first, pbp.first.first);
      auto kit1 = citerator(lmap, pbp.first.second, pbp.first.second);
      if (kit0->second > kit1->second) { std::swap(kit0, kit1); }
      order.emplace_back(kit0->second, kit1->second, keyToAssign);
      kit0->second = keyToAssign;
      kit1->second = keyToAssign;
      ++keyToAssign;
    }
  }
}


// template <typename TVal, typename TKey, typename TTr> void
// genContourConfidenceMap (
//     std::map<std::vector<TKey>, TVal>& pcmap,
//     std::vector<TTr> const& trees,
//     std::vector<std::vector<std::vector<TKey>>> const& subKeys,
//     std::vector<std::vector<int>> const& picks)
// {

// }


// template <typename TVal, typename TRegionMap, typename TTr>
// genContourConfidenceMap (
//     std::unordered_map<std::pair<int, int>, TVal>& pcmap,
//     std::vector<TRegionMap> const& rmaps,
//     std::vector<TTr> const& trees,
//     std::vector<std::vector<int>> const& picks,
//     std::vector<std::vector<std::vector<TKey>>> const& subKeys)
// {
//   typedef std::unordered_map<std::pair<int, int>, TVal> PcMap;
//   std::map<std::vector<TKey>, PcMap::iterator> ancestorMap;
//   int nTree = trees.size();
//   for (int i = 0; i < nTree; ++i) {
//     for (int pi : picks[i]) {
//       int nSelfOrDescendants = 1;
//       for (auto& pc : pcmap) {
//         if (subKeys[pc.first.first][pc.first.second] == subKeys[i][pi]) {
//           ++pc.second;
//         } else if ()
//       }
//     }
//   }
// }

};
};

#endif
