#ifndef _glia_hmt_tree_fix_hxx_
#define _glia_hmt_tree_fix_hxx_

#include "type/tree.hxx"

namespace glia {
namespace hmt {

// Tags:
//   0: untouched
//   1: strong itself
//   2: weak itself w/ strong descendant(s)
//   3: weak itself w/o strong descendant(s)
// f(node): determine if node is weak
template <typename TTree, typename Func> void
getWeakRoots (std::vector<int>& weakRoots, TTree const& tree, Func f)
{
  std::vector<int> tags;
  int n = tree.size();
  tags.resize(n, 0);
  for (int i = 0; i < n; ++i) {
    auto const& node = tree[i];
    if (f(node)) { // Weak itself
      tags[i] = node.isLeaf() || std::all_of
          (node.children.begin(), node.children.end(),
           [&tags](int j){ return tags[j] == 3; })? 3: 2;

    }
    else { tags[i] = 1; } // Strong itself
  }
  std::queue<int> q;
  q.push(tree.root());
  while (!q.empty()) {
    int i = q.front();
    q.pop();
    if (tags[i] == 3) // Weak itself w/o strong descedant(s)
      // { if (tree.height(i) > 1) { weakRoots.push_back(i); } }
    { weakRoots.push_back(i); }
    else if (tags[i] == 2) // Weak itself w/ strong descendant(s)
    { for (auto child: tree[i].children) { q.push(child); } }
  }
}

};
};

#endif
