#ifndef _glia_hmt_tree_resolve_greedy_hxx_
#define _glia_hmt_tree_resolve_greedy_hxx_

#include "hmt/tree_build.hxx"

namespace glia {
namespace hmt {

template <typename TTree, typename VFunc, typename CFunc> int
pickTreeNode (TTree const& tree, VFunc fvalid, CFunc fcomp)
{
  int ret = -1;
  for (auto const& node : tree) {
    if (fvalid(node) && (ret < 0 || fcomp(tree[ret], node)))
    { ret = node.self; }
  }
  return ret;
}


template <typename TContainer, typename TTree, typename VFunc,
          typename CFunc, typename PFunc> void
resolveTreeGreedy (
    TContainer& picks, TTree const& tree, VFunc fvalid, CFunc fcomp,
    PFunc fproc)
{
  picks.reserve(tree.size());
  int pi = pickTreeNode(tree, fvalid, fcomp);
  while (pi >= 0) {
    picks.push_back(pi);
    fproc(tree[pi]);
    pi = pickTreeNode(tree, fvalid, fcomp);
  }
}


template <typename TTree, typename Func> int
pickTreeNode (TTree const& tree, std::vector<bool> const& validity,
              Func comp)
{
  int ret = -1;
  for (auto const& node: tree) {
    if (validity[node.self] && (ret < 0 || comp(tree[ret], node)))
    { ret = node.self; }
  }
  return ret;
}


// fvalid(node): returns whether a node is valid initially
template <typename TContainer, typename TTree, typename VFunc,
          typename CFunc> void
resolveTreeGreedy (TContainer& picks, TTree const& tree, VFunc fvalid,
                   CFunc comp)
{
  picks.reserve(tree.size());
  std::vector<bool> validity(tree.size());
  for (auto const& node : tree) { validity[node.self] = fvalid(node); }
  int pi = pickTreeNode(tree, validity, comp);
  while (pi >= 0) {
    picks.push_back(pi);
    validity[pi] = false;
    tree.traverseAncestors(pi, [&validity]
                           (typename TTree::Node const& node)
                           { validity[node.self] = false; });
    tree.traverseDescendants(pi, [&validity]
                             (typename TTree::Node const& node)
                             { validity[node.self] = false; });
    pi = pickTreeNode(tree, validity, comp);
  }
}


// Do not require node.data.label
template <typename TContainer, typename TTree, typename Func> void
resolveTreeGreedy (TContainer& picks, TTree const& tree, Func comp)
{ resolveTreeGreedy(picks, tree, f_true<typename TTree::Node>, comp); }



// Return (treeIndex, nodeIndex)
template <typename TTree, typename Func> std::pair<int, int>
pickTreeNode (std::vector<TTree> const& trees,
              std::vector<std::vector<bool>> const& validity,
              Func comp)
{
  auto ret = std::make_pair(-1, -1);
  int nTree = trees.size();
  for (int i = 0; i < nTree; ++i) {
    for (auto const& node: trees[i]) {
      if (validity[i][node.self] &&
          (ret.first < 0 ||
           comp(trees[ret.first][ret.second], node))) {
        ret.first = i;
        ret.second = node.self;
      }
    }
  }
  return ret;
}


// Require node.data.label
template <typename TTree, typename Func> void
resolveTreeGreedy (std::vector<std::pair<int, int>>& picks,
                   std::vector<TTree> const& trees, Func comp)
{
  typedef decltype(trees.front().front().data.label) Key;
  picks.reserve(trees.front().size()); // Too big??
  int nTree = trees.size();
  std::vector<std::vector<bool>> validity(nTree);
  std::vector<std::unordered_map<Key, int>> lnmap(nTree);
  for (int i = 0; i < nTree; ++i) {
    validity[i].resize(trees[i].size(), true);
    trees[i].traverseLeaves
        (trees[i].root(), [&lnmap, i](typename TTree::Node const& tn)
         { lnmap[i][tn.data.label] = tn.self; });
  }
  std::vector<Key> llabels;
  llabels.reserve(trees.front().size()); // Too big??
  auto pick = pickTreeNode(trees, validity, comp);
  while (pick.first >= 0) {
    picks.push_back(pick);
    validity[pick.first][pick.second] = false;
    trees[pick.first].traverseAncestors
        (pick.second, [&validity, &pick]
         (typename TTree::Node const& node)
         { validity[pick.first][node.self] = false; });
    llabels.clear();
    trees[pick.first].traverseDescendants
        (pick.second, [&validity, &pick, &llabels]
         (typename TTree::Node const& node) {
          validity[pick.first][node.self] = false;
          if (node.isLeaf()) { llabels.push_back(node.data.label); }
        });
    for (auto llabel: llabels) {
      for (int i = 0; i < nTree; ++i) {
        if (i != pick.first) {
          auto nit = lnmap[i].find(llabel);
          if (nit != lnmap[i].end()) {
            validity[i][nit->second] = false;
            trees[i].traverseAncestors
                (nit->second, [&validity, i]
                 (typename TTree::Node const& node)
                 { validity[i][node.self] = false; });
          }
        }
      }
    }
    pick = pickTreeNode(trees, validity, comp);
  }
}


template <typename TTree, typename Func> void
resolveTreeGreedy (
    std::vector<std::vector<int>>& picks,
    std::vector<TTree> const& trees, Func comp)
{
  int nTree = trees.size();
  typedef decltype(trees.front().front().data.label) Key;
  // Prepare subKeys
  std::vector<std::vector<std::vector<Key>>> subKeys(nTree);
  for (int i = 0; i < nTree; ++i) {
    collectSubKeys(
        subKeys[i], trees[i], [](typename TTree::Node const& node)
        { return node.data.label; }, true);
  }
  // Prepare auxiliary variables
  picks.resize(nTree);
  std::vector<std::vector<bool>> validity(nTree);
  std::vector<std::unordered_map<Key, int>> lnmap(nTree);
  for (int i = 0; i < nTree; ++i) {
    validity[i].resize(trees[i].size(), true);
    trees[i].traverseLeaves(
        trees[i].root(), [&lnmap, i](typename TTree::Node const& tn)
        { lnmap[i][tn.data.label] = tn.self; });
    picks[i].reserve(trees[i].size());
  }
  // Resolve
  std::vector<Key> llabels;
  llabels.reserve(trees.front().size());
  std::set<int> nodeIndices;
  auto pick = pickTreeNode(trees, validity, comp);
  while (pick.first >= 0) {
    picks[pick.first].push_back(pick.second);
    validity[pick.first][pick.second] = false;
    trees[pick.first].traverseAncestors
        (pick.second, [&validity, &pick]
         (typename TTree::Node const& node)
         { validity[pick.first][node.self] = false; });
    llabels.clear();
    trees[pick.first].traverseDescendants
        (pick.second, [&validity, &pick, &llabels]
         (typename TTree::Node const& node) {
          validity[pick.first][node.self] = false;
          if (node.isLeaf()) { llabels.push_back(node.data.label); }
        });
    for (int i = 0; i < nTree; ++i) {
      if (i != pick.first) {
        nodeIndices.clear();
        for (auto const& ll : llabels) {
          auto nit = lnmap[i].find(ll);
          nodeIndices.insert(nit->second);
          trees[i].traverseAncestors(
              nit->second, [&nodeIndices, &validity, i](
                  typename TTree::Node const& node) {
                if (validity[i][node.self])
                { nodeIndices.insert(node.self); }
              });
        }
        for (auto niit = nodeIndices.rbegin();
             niit != nodeIndices.rend(); ++niit) {
          if (validity[i][*niit] && std::includes(
                  subKeys[pick.first][pick.second].begin(),
                  subKeys[pick.first][pick.second].end(),
                  subKeys[i][*niit].begin(), subKeys[i][*niit].end())) {
            picks[i].push_back(*niit);
            validity[i][*niit] = false;
            trees[i].traverseDescendants(
                *niit, [&validity, i](typename TTree::Node const& node) {
                  validity[i][node.self] = false;
                });
          } else { validity[i][*niit] = false; }
        }
      }
    }
    pick = pickTreeNode(trees, validity, comp);
  }
}

};
};

#endif
