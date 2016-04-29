#ifndef _glia_alg_tree_hxx_
#define _glia_alg_tree_hxx_

#include "glia_base.hxx"

namespace glia {
namespace alg {

template <typename TTree, typename Func> void
bfs (TTree& tree, int root, Func f) {
  if (root < 0) { return; }
  std::queue<typename TTree::Node*> q;
  q.push(&tree[root]);
  while (!q.empty()) {
    auto pn = q.front();
    q.pop();
    f(*pn);
    for (auto c: pn->children) { q.push(&tree[c]); }
  }
}


template <typename TTree, typename Func> void
bfs (TTree const& tree, int root, Func f) {
  if (root < 0) { return; }
  std::queue<typename TTree::Node const*> q;
  q.push(&tree[root]);
  while (!q.empty()) {
    auto pn = q.front();
    q.pop();
    f(*pn);
    for (auto c: pn->children) { q.push(&tree[c]); }
  }
}


// f(node): returns identifier for leaf nodes
template <typename TKey, typename TTree, typename Func> void
encode (std::vector<TKey>& code, TTree const& tree, Func f)
{
  int n = tree.size();
  code.reserve(code.size() + n * n * 3);
  std::vector<bool> processed(n, false);
  std::vector<TKey> keys(n);
  std::deque<std::pair<TKey, int>> nodeQueue;
  tree.traverseLeaves(tree.root(), [&processed, &keys, &nodeQueue, &f]
                      (typename TTree::Node const& node)
                      {
                        TKey key = f(node);
                        keys[node.self] = key;
                        nodeQueue.emplace_back(key, node.self);
                        processed[node.self] = true;
                      });
  std::sort(nodeQueue.begin(), nodeQueue.end());
  // nodeQueue.sort();
  TKey keyToAssign = nodeQueue.back().first + 1;
  while (!nodeQueue.empty()) {
    auto self = nodeQueue.front().second;
    auto pa = tree[self].parent;
    if (pa >= 0) { // Non-root
      if (processed[self] && !processed[pa]) {
        bool siblingAllProcessed = true;
        for (auto child: tree[pa].children) {
          if (child != self && !processed[child]) {
            siblingAllProcessed = false;
            break;
          }
        }
        if (siblingAllProcessed) {
          std::vector<TKey> siblingKeys;
          tree.traverseSiblings
              (self, [&siblingKeys, &keys]
               (typename TTree::Node const& tn)
               { siblingKeys.push_back(keys[tn.self]); });
          std::sort(siblingKeys.begin(), siblingKeys.end());
          keys[pa] = keyToAssign++;
          nodeQueue.emplace_back(keys[pa], pa);
          processed[pa] = true;
          code.push_back(keys[self]);
          splice(code, siblingKeys);
          code.push_back(keys[pa]);
        }
        else { nodeQueue.emplace_back(nodeQueue.front()); }
      }
    }
    else if (processed[self]) { code.push_back(keys[self]); } // Root
    nodeQueue.pop_front();
  }
}

};
};

#endif
