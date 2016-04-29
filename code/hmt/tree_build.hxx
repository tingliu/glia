#ifndef _glia_hmt_tree_build_hxx_
#define _glia_hmt_tree_build_hxx_

#include "type/tuple.hxx"
#include "type/tree.hxx"

namespace glia {
namespace hmt {

// f: tree node initialization function
// void f (TTreeNode& node, TKey nodeLabel)
template <typename TTree, typename TKey, typename Func> void
genTree (TTree& tree, std::vector<TTriple<TKey>> const& order, Func f)
{
  tree.reserve(order.size() * 2 + 1);
  std::unordered_map<TKey, int> nmap;
  int ni = 0;
  for (auto const& merge: order) {
    auto nit0 = nmap.find(merge.x0);
    auto nit1 = nmap.find(merge.x1);
    if (nit0 == nmap.end()) {
      tree.emplace_back(ni, -1);
      f(tree.back(), merge.x0);
      nit0 = nmap.emplace(merge.x0, ni++).first;
    }
    if (nit1 == nmap.end()) {
      tree.emplace_back(ni, -1);
      f(tree.back(), merge.x1);
      nit1 = nmap.emplace(merge.x1, ni++).first;
    }
    tree[nit0->second].parent = ni;
    tree[nit1->second].parent = ni;
    tree.emplace_back
        (ni, -1, std::initializer_list<int>{nit0->second, nit1->second});
    f(tree.back(), merge.x2);
    nmap.emplace(merge.x2, ni++);
  }
}


template <typename TTree, typename TKey>
std::vector<double>::const_iterator
genTreeWithNodePotentials
(TTree& tree, std::vector<TTriple<TKey>> const& order,
 std::vector<double>::const_iterator mpit)
{
  genTree(tree, order, [&mpit, &tree]
          (typename TTree::Node& node, TKey r) {
            node.data.label = r;
            if (!node.isLeaf()) {
              node.data.potential = *mpit;
              double pSplit = 1.0 - *mpit;
              for (auto child: node.children) {
                if (tree[child].isLeaf())
                { tree[child].data.potential = pSplit * pSplit; }
                else { tree[child].data.potential *= pSplit; }
              }
              ++mpit;
            }
          });
  tree[tree.root()].data.potential *= tree[tree.root()].data.potential;
  return mpit;
}


template <typename TTree, typename TKey> void
genOrder (std::vector<TTriple<TKey>>& order, TTree const& tree)
{
  int n = tree.size();
  order.reserve(order.size() + n / 2);
  for (auto const& node: tree) {
    if (!node.isLeaf()) {
      order.emplace_back(tree[node.children.front()].data.label,
                         tree[node.children.back()].data.label,
                         node.data.label);
    }
  }
}


// template <typename TTree, typename TKey> void
// genJungle (std::vector<std::vector<std::pair<int, int>>>& jungle,
//            std::vector<TTree> const& trees)
// {
//   typedef std::vector<TKey> LKey; // Leaf key
//   int nTree = trees.size();
//   std::vector<std::vector<LKey>> leafKeys(nTree);
//   int nTreeNode = 0; // # of all tree nodes
//   for (int i = 0; i < nTree; ++i) {
//     int n = trees[i].size();
//     nTreeNode += n;
//     leafKeys[i].resize(n);
//     for (int j = 0; j < n; ++j) {
//       if (trees[i][j].isLeaf())
//         { leafKeys[i][j].push_back(trees[i][j].data.label); }
//     }
//   }
//   jungle.reserve(n); // Too big??
//   // {leafLabels} -> jungleNodeIndex
//   std::map<LKey, int> lsmap(compare);
// }


// f(node) should return node key
template <typename TTree, typename TKey, typename Func> void
collectSubKeys (
    std::vector<std::vector<TKey>>& subKeys, TTree const& tree,
    Func f, bool sortKeys)
{
  subKeys.reserve(tree.size());
  for (auto const& node : tree) {
    subKeys.push_back(std::vector<TKey>());
    if (node.isLeaf()) { subKeys.back().push_back(f(node)); } else {
      for (auto const& c : node.children)
      { append(subKeys.back(), subKeys[c]); }
    }
    if (sortKeys) {
      std::sort(subKeys.back().begin(), subKeys.back().end());
    }
  }
}


template <typename TKey> void
genMergePaths (std::vector<std::vector<int>>& mergePaths,
               std::vector<TTriple<TKey>> const& order)
{
  int n = order.size();
  std::unordered_map<TKey, int> nonLeafNodes;
  std::unordered_map<TKey, int> childMergeIndices;
  for (int i = 0; i < n; ++i) {
    childMergeIndices[order[i].x0] = i;
    childMergeIndices[order[i].x1] = i;
    nonLeafNodes[order[i].x2] = i;
    if (nonLeafNodes.count(order[i].x0) == 0 &&
        nonLeafNodes.count(order[i].x1) == 0) {
      mergePaths.push_back(std::vector<int>(1, i));
    }
  }
  for (auto& path : mergePaths) {
    auto it = childMergeIndices.find(order[path.back()].x2);
    while (it != childMergeIndices.end()) {
      path.push_back(it->second);
      it = childMergeIndices.find(order[it->second].x2);
    }
  }
}


template <typename TKey> void
genMergePaths (std::vector<std::vector<int>>& mergePaths,
               std::vector<TTriple<TKey>> const& order,
               int pathLength, int minPathLength)
{
  int n = order.size();
  std::unordered_map<TKey, int> nonLeafNodes;
  std::unordered_map<TKey, int> childMergeIndices;
  std::vector<std::vector<int>> allPaths;
  allPaths.reserve(n);
  for (int i = 0; i < n; ++i) {
    childMergeIndices[order[i].x0] = i;
    childMergeIndices[order[i].x1] = i;
    nonLeafNodes[order[i].x2] = i;
    allPaths.push_back(std::vector<int>(1, i));
  }
  for (auto& path : allPaths) {
    auto it = childMergeIndices.find(order[path.back()].x2);
    while (it != childMergeIndices.end() && path.size() < pathLength) {
      path.push_back(it->second);
      it = childMergeIndices.find(order[it->second].x2);
    }
    if (path.size() == pathLength ||
        (path.size() >= minPathLength &&
         nonLeafNodes.count(order[path.front()].x0) == 0 &&
         nonLeafNodes.count(order[path.front()].x1) == 0)) {
      mergePaths.emplace_back();
      mergePaths.back().swap(path);
    }
  }
}


template <typename TTree> void
genNodePaths (std::vector<std::vector<int>>& nodePaths,
              TTree const& tree)
{
  for (auto const& node : tree) {
    if (node.isLeaf()) {
      nodePaths.push_back(std::vector<int>(1, node.self));
      tree.traverseAncestors(
          node.self, [&nodePaths](typename TTree::Node const& n) {
            nodePaths.back().push_back(n.self);
          });
    }
  }
}

};
};

#endif
