#ifndef _glia_hmt_tree_ccm_hxx_
#define _glia_hmt_tree_ccm_hxx_

#include "type/tree.hxx"
#include "util/stats.hxx"
#include "alg/combinatorics.hxx"

namespace glia {
namespace hmt {

// Compute energy tuple using dynamic programming (bottom-up)
template <typename TTree> void
computeEnergyTuples (std::vector<std::pair<double, double>>& Ems,
                     TTree const& tree)
{
  uint n = tree.size();
  Ems.resize(n);
  for (int i = 0; i < n; ++i) {
    Ems[i].first = tree[i].data.Em;
    Ems[i].second = tree[i].data.Es;
    for (auto child: tree[i].children) {
      stats::plusEqual(Ems[i].first, Ems[child].first);
      stats::plusEqual
          (Ems[i].second, std::min(Ems[child].first, Ems[child].second));
    }
  }
}


// Resolve factor tree given energy tuple (top-down)
template <typename TContainer, typename TTree> void
resolveFactorTree
(TContainer& picks, TTree const& tree,
 std::vector<std::pair<double, double>> const& Ems)
{
  picks.reserve(tree.size());
  std::queue<int> nq;
  nq.push(tree.root());
  while (!nq.empty()) {
    int i = nq.front();
    nq.pop();
    if (Ems[i].first < Ems[i].second) { picks.push_back(i); }
    else {
      for (auto child: tree[i].children) { nq.push(child); }
    }
  }
}


template <typename TContainer, typename TTree> void
resolveFactorTree (TContainer& picks, TTree const& tree)
{
  // Bottom-up compute energy tuple
  std::vector<std::pair<double, double>> Ems;
  computeEnergyTuples(Ems, tree);
  // Top-down pick nodes
  resolveFactorTree(picks, tree, Ems);
}


// Compute best labeling energy for node i being labeled segment
template <typename TTree> double
computeFactorNodeEnergyPositive
(TTree const& tree, int i,
 std::vector<std::pair<double, double>> const& Ems)
{
  double ret = Ems[i].first; // All descendants merge
  int prevChild = i;
  tree.traverseAncestors
      (i, [&ret, &prevChild, &Ems](typename TTree::Node const& node)
       {
         stats::plusEqual(ret, node.data.Es);
         for (auto child: node.children) {
           if (child != prevChild) {
             stats::plusEqual
                 (ret, std::min(Ems[child].first, Ems[child].second));
           }
         }
         prevChild = node.self;
       });
  return ret;
}


// Compute best labeling energy for node i being labeled not segment
template <typename TTree> double
computeFactorNodeEnergyNegative
(TTree const& tree, int i,
 std::vector<std::pair<double, double>> const& Ems)
{
  double ret = tree[i].data.Es;
  std::queue<int> nq;
  for (auto child: tree[i].children) { nq.push(child); }
  while (!nq.empty()) {
    int j = nq.front();
    nq.pop();
    if (Ems[j].first < Ems[j].second)
    { stats::plusEqual(ret, Ems[j].first); }
    else { for (auto child: tree[j].children) { nq.push(child); } }
  }
  int prevChild = i;
  tree.traverseAncestors
      (i, [&ret, &prevChild, &Ems](typename TTree::Node const& node)
       {
         stats::plusEqual(ret, node.data.Es);
         for (auto child: node.children) {
           if (child != prevChild) {
             stats::plusEqual
                 (ret, std::min(Ems[child].first, Ems[child].second));
           }
         }
         prevChild = node.self;
       });
  return ret;
}


// Compute all possible energies for every node
// Ems[i]: (Em, {Es})
// Em: All nodes merge below
// Es: Splits happen below
// Caution: exponential time/space complexity!
template <typename TTree> void
computeFactorTreeFullLabelEnergy
(std::vector<std::pair<double, std::vector<double>>>& Ems,
 TTree const& tree)
{
  uint n = tree.size();
  Ems.resize(n);
  for (int i = 0; i < n; ++i) {
    auto const& node = tree[i];
    if (node.isLeaf()) {
      Ems[i].first = 0.0;
      Ems[i].second.push_back(0.0);
    }
    else {
      double Em = 0.0;
      uint nEs = 1;
      std::vector<std::vector<double> const*> pEss;
      pEss.reserve(node.children.size());
      for (auto child: node.children) {
        stats::plusEqual(Em, Ems[child].first);
        pEss.push_back(&Ems[child].second);
        nEs *= Ems[child].second.size();
      }
      Ems[i].first = node.data.Em;
      stats::plusEqual(Ems[i].first, Em);
      ++nEs;
      Ems[i].second.reserve(nEs);
      alg::combination(Ems[i].second, node.data.Es, pEss, pEss.begin(),
                       [](double& res, double x)
                       { stats::plusEqual(res, x); });
      Ems[i].second.push_back(node.data.Em);
      stats::plusEqual(Ems[i].second.back(), Em);
    }
  }
}


// Marginal energy for node i being labeled true segment
// Ems is computed using computeFactorTreeFullLabelEnergy()
// Caution: exponential time/space complexity!
template <typename TTree> void
computeFactorNodeMarginalEnergy
(std::vector<double>& Emargin, TTree const& tree, int i,
 std::vector<std::pair<double, std::vector<double>>> const& Ems)
{
  double Ebranch = tree[i].data.Es; // Energy on this branch
  for (auto child: tree[i].children)
  { stats::plusEqual(Ebranch, Ems[child].first); }
  int prevChild = i;
  std::vector<std::vector<double> const*> pEss;
  uint nEs = 1;
  tree.traverseAncestors(i, [&Ems, &Ebranch, &prevChild, &pEss, &nEs]
                         (typename TTree::Node const& node)
                         {
                           stats::plusEqual(Ebranch, node.data.Es);
                           for (auto child: node.children) {
                             if (child != prevChild) {
                               pEss.push_back(&Ems[child].second);
                               nEs += Ems[child].second.size();
                             }
                           }
                           prevChild = node.self;
                         });
  Emargin.reserve(nEs);
  alg::combination(Emargin, Ebranch, pEss, pEss.begin(),
                   [](double& res, double x)
                   { stats::plusEqual(res, x); });
}


// namespace {

// // Example: R1 + R2 -> R3
// //   Em(R3) = Em(R1) + Em(R2) + Em(R1 + R2)
// //   Es(R3) = min(Em(R1) + Es(R1)) + min(Em(R2) + Es(R2)) + Es(R1 + R2)
// // in which:
// //   Em(Ri) = minEm(Rj), s.t. Rj = Ri (same subKeys)
// //   Es(Ri) = minEs(Rj), s.t. Rj = Ri (same subKeys)
// template <typename TTree, typename TKey> void
// helperComputeEnergyTuples (
//     std::map<std::vector<TKey>, std::pair<double, double>>& bestScores,
//     std::vector<std::vector<std::pair<double, double>>>& Ems,
//     std::vector<TTree> const& trees, int treeIndex, int nodeIndex,
//     std::vector<std::vector<TKey>> const& subKeys,
//     std::vector<std::map<std::vector<TKey>, int>> const& nodeIndexMap)
// {
//   auto sit = bestScores.find(subKeys[treeIndex][nodeIndex]);
//   if (sit != bestScores.end()) {
//     Ems[treeIndex][nodeIndex] = sit->second;
//     return;
//   }
//   TTree const& node = trees[treeIndex][nodeIndex];
//   int nc = node.children.size();
//   double Em = node.data.Em, Es = node.data.Es;
//   for (int c : node.children) {
//     auto const& subKey = subKeys[treeIndex][c];
//     double minEm = FMAX, minEs = FMAX;
//     for (int i = 0; i < trees.size(); ++i) {
//       auto kit = nodeIndexMap[i].find(subKey);
//       if (kit != nodeIndexMap[i].end()) {
//         helperComputeEnergyTuples(
//             bestScores, Ems, trees, i, kit->second,
//             subKeys, nodeIndexMap);
//         minEm = std::min(minEm, Ems[i][kit->second].first);
//         minEs = std::min(minEs, Ems[i][kit->second].second);
//       }
//     }
//     bestScores[subKey] = std::make_pair(minEm, minEs);
//     stats::plusEqual(Em, minEm);
//     stats::plusEqual(Es, std::min(minEm, minEs));
//   }
//   Ems[treeIndex][nodeIndex].first = Em;
//   Ems[treeIndex][nodeIndex].second = Es;
// }

// };


// template <typename TKey, typename TTree> void
// computeEnergyTuples (
//     std::vector<std::vector<std::pair<double, double>>>& Ems,
//     std::vector<TTree> const& trees,
//     std::vector<std::vector<std::vector<TKey>>> const& subKeys,
//     std::vector<std::map<std::vector<TKey>, int>> const& nodeIndexMap)
// {
//   std::map<std::vector<TKey>, EnergyTuple> bestScores;
//   int nTree = trees.size();
//   // Compute
//   for (int i = 0; i < nTree; ++i) {
//     helperComputeEnergyTuples(
//         bestScores, Ems, trees, i, trees[i].root(), subKeys,
//         nodeIndexMap);
//   }
// }


// namespace {

// struct EnergyTupleToSort {
//   int treeIndex;
//   int nodeIndex;
//   double Em;
//   double Es;

//   EnergyTupleToSort (
//       int treeIndex_, int nodeIndex_, double Em_, double Es_)
//       : treeIndex(treeIndex_), nodeIndex(nodeIndex_), Em(Em_), Es(Es_) {}

//   bool operator< (EnergyTupleToSort const& rhs) const {
//     return std::min(Em, Es) > std::min(rhs.Em, rhs.Es);
//   }

//   bool isMerge () const { return Em < Es; }
// };

// };


// template <typename TKey, typename TTree> void
// resolveFactorTree (
//     std::vector<std::vector<int>>& picks, std::vector<TTree> const& trees,
//     std::vector<std::vector<std::pair<double, double>>> const& Ems,
//     std::vector<std::vector<std::vector<TKey>>> const& subKeys,
//     std::vector<std::map<std::vector<TKey>, int>> const& nodeIndexMap)
// {
//   int nTree = trees.size();
//   picks.resize(nTree);
//   std::vector<std::vector<bool>> validty(nTree);
//   std::vector<std::unordered_map<TKey, int>> lnmap(nTree);
//   std::priority_queue<EnergyTupleToSort> q;
//   for (int i = 0; i < nTree; ++i) {
//     int nNode = trees[i].size();
//     picks[i].reserve(nNode);
//     validty[i].resize(nNode, true);
//     for (int j = 0; j < nNode; ++j) {
//       q.push(EnergyTupleToSort(i, j, Ems[i][j].first, Ems[i][j].second));
//       auto const& node = trees[i][j];
//       if (node.isLeaf()) { lnmap[i][node.data.label] = node.self; }
//     }
//   }
//   std::vector<TKey> llabels;
//   llabels.reserve(trees.front().size());
//   while (!q.empty()) {
//     auto const& et = q.top();
//     if (et.isMerge) {  // Merge: only consistent merges will be allowed
//       if (validty[et.treeIndex][et.nodeIndex]) {
//         // Pick
//         picks[et.treeIndex].push_back(et.nodeIndex);
//         llabels.clear();
//         // Invalidate all descendants in this tree and collect leaf nodes
//         trees[et.treeIndex].traverseDescendants(
//             et.nodeIndex, [&et, &llabels](
//                 typename TTree::Node const& node) {
//               validty[et.treeIndex][node.self] = false;
//               if (node.isLeaf()) { llabels.push_back(node.data.label); }
//             });
//         // Invalidate inconsistent merges in other trees
//         for (int i = 0; i < nTree; ++i) {
//           if (i != et.treeIndex) {
//             for (llabel: llabels) {
//               auto nit = lnmap[i].find(llabel);
//               if (nit != lnmap[i].end()) {
//                 // Whether to invalidate all ancestors
//                 bool setAllAncestorsFalse = false;
//                 for (int ii = trees[i][*nit].parent; ii >= 0;
//                      ii = trees[i][ii].parent) {
//                   if (validty[i][ii]) {
//                     if (setAllAncestorsFalse) {
//                       validty[i][ii] = false;
//                     } else if (!std::includes(
//                         subKeys[et.treeIndex][et.nodeIndex].begin(),
//                         subKeys[et.treeIndex][et.nodeIndex].end(),
//                         subKeys[i][ii].begin(),
//                         subKeys[i][ii].end())) {
//                       // Invalidate consistent node, and all its ancesotrs
//                       validty[i][ii] = false;
//                       setAllAncestorsFalse = true;
//                     } else if (subKeys[et.treeIndex][et.nodeIndex] ==
//                                subKeys[i][ii]) {
//                       // If same node as picked node is found
//                       // Pick it and invalidate all its ancestors
//                       picks[i].push_back(ii);
//                       validty[i][ii] = false;
//                       trees[i].traverseDescendants(
//                           ii, [&validty, i](
//                               typename TTree::Node const& node) {
//                             validty[i][node.self] = false;
//                           });
//                       setAllAncestorsFalse = true;
//                     }
//                   } else { break; }
//                 }
//               }
//             }
//           }
//         }
//       }
//     } else {  // Split: no merge is allowed at this (or same) node
//       for (int i = 0; i < nTree; ++i) {
//         auto niit =
//             nodeIndexMap[i].find(subKeys[et.treeIndex][et.nodeIndex]);
//         if (niit != nodeIndexMap[i].end()) { validty[i][*niit] = false; }
//       }
//     }
//     validty[et.treeIndex][et.nodeIndex] = false;
//     q.pop();
//   }
// }

};
};

#endif
