#include "hmt/bc_label.hxx"
#include "hmt/tree_build.hxx"
#include "type/big_num.hxx"
#include "type/tuple.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/image_stats.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

std::string bcLabelFile;
std::string segImageFile;
std::string mergeOrderFile;
std::string truthImageFile;
std::string maskImageFile;
bool usePairF1 = true;
bool optSplit = false;
int globalOpt = 0;  // 0: Bypass, 1: RCC by merge, 2: RCC by split
bool tweak = false;
double maxPrecDrop = 1.0;

struct NodeData {
  Label label;
  std::vector<int> bestSplits;
  int bcLabel = BC_LABEL_MERGE;
};

bool operation ()
{
  std::vector<TTriple<Label>> order;
  readData(order, mergeOrderFile, true);
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto truthImage = readImage<LabelImage<DIMENSION>>(truthImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, order, false);
  int n = order.size();
  std::vector<int> bcLabels;
  if (globalOpt == 0) {  // Local optimal
    bcLabels.resize(n);
    if (usePairF1) {  // Use pair F1
      std::vector<double> mergeF1s, splitF1s;
      mergeF1s.resize(n);
      splitF1s.resize(n);
      parfor(0, n, true, [&bcLabels, &mergeF1s, &splitF1s, &rmap, &order,
                          &truthImage](int i) {
               bcLabels[i] = genBoundaryClassificationLabelF1<BigInt>(
                   mergeF1s[i], splitF1s[i], rmap, order[i].x0,
                   order[i].x1, order[i].x2, truthImage, tweak,
                   maxPrecDrop);
             }, 0);
      if (optSplit) {
        std::unordered_map<Label, std::vector<RegionMap::Region const*>>
            smap;
        for (int i = 0; i < n; ++i) {
          auto sit0 = citerator(
              smap, order[i].x0, 1, &rmap.find(order[i].x0)->second);
          auto sit1 = citerator(
              smap, order[i].x1, 1, &rmap.find(order[i].x1)->second);
          std::vector<RegionMap::Region const*> pSplitRegions,
              pMergeRegions{&rmap.find(order[i].x2)->second};
          append(pSplitRegions, sit0->second, sit1->second);
          if (bcLabels[i] == BC_LABEL_SPLIT)  // Already split - skip
          { splice(smap[order[i].x2], pSplitRegions); }
          else {  // Merge - check
            double optSplitF1, optSplitPrec, optSplitRec;
            stats::pairF1<BigInt>(optSplitF1, optSplitPrec, optSplitRec,
                                  pSplitRegions, truthImage, {BG_VAL});
            if (mergeF1s[i] > optSplitF1)
            { splice(smap[order[i].x2], pMergeRegions); }
            else {
              bcLabels[i] = BC_LABEL_SPLIT;
              splice(smap[order[i].x2], pSplitRegions);
            }
          }
        }
      }
    } else {  // Use traditional Rand index
      parfor(0, n, true, [&bcLabels, &rmap, &order, &truthImage](int i) {
          double mergeRI, splitRI;
          bcLabels[i] = genBoundaryClassificationLabelRI<BigInt>(
              mergeRI, splitRI, rmap, order[i].x0, order[i].x1,
              order[i].x2, truthImage);
        }, 0);
    }
  } else {  // Global optimal
    typedef TTree<NodeData> Tree;
    Tree tree;
    genTree(tree, order, [](Tree::Node& node, Label r) {
        node.data.label = r; });
    for (auto& node : tree) {
      if (node.isLeaf()) {
        node.data.bcLabel = BC_LABEL_MERGE;
        node.data.bestSplits.push_back(node.self);
      } else {
        std::vector<RegionMap::Region const*>
            pMergeRegions{&rmap.find(node.data.label)->second};
        std::vector<RegionMap::Region const*> pSplitRegions;
        for (auto c : node.children) {
          for (auto j : tree[c].data.bestSplits) {
            pSplitRegions.push_back(
                &rmap.find(tree[j].data.label)->second);
          }
        }
        double mergeF1 = stats::pairF1<BigInt>(
            pMergeRegions, truthImage, {BG_VAL});
        double splitF1 = stats::pairF1<BigInt>(
            pSplitRegions, truthImage, {BG_VAL});
        if (mergeF1 > splitF1) {
          node.data.bcLabel = BC_LABEL_MERGE;
          node.data.bestSplits.push_back(node.self);
        } else {
          node.data.bcLabel = BC_LABEL_SPLIT;
          for (auto c : node.children) {
            for (auto j : tree[c].data.bestSplits)
            { node.data.bestSplits.push_back(j); }
          }
        }
      }
    }
    if (globalOpt == 1) {
      // Enforce path consistency by merging
      std::queue<int> q;
      q.push(tree.root());
      while (!q.empty()) {
        int ni = q.front();
        q.pop();
        if (tree[ni].data.bcLabel == BC_LABEL_MERGE) {
          tree.traverseDescendants(ni, [](Tree::Node& node) {
              node.data.bcLabel = BC_LABEL_MERGE; });
        } else { for (auto c : tree[ni].children) { q.push(c); } }
      }
    } else if (globalOpt == 2) {
      // Enforce path consistency by splitting
      for (auto const& node : tree) {
        if (node.data.bcLabel == BC_LABEL_SPLIT) {
          tree.traverseAncestors(node.self, [](Tree::Node& tn) {
              tn.data.bcLabel = BC_LABEL_SPLIT;
            });
        }
      }
    } else { perr("Error: unsupported globalOpt type..."); }
    // Collect bcLabels
    bcLabels.reserve(n);
    for (auto const& node : tree)
    { if (!node.isLeaf()) { bcLabels.push_back(node.data.bcLabel); } }
  }
  if (!bcLabelFile.empty()) { writeData(bcLabelFile, bcLabels, "\n"); }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("truthImage,t", bpo::value<std::string>(&truthImageFile)->required(),
       "Input ground truth segmentation image file name")
      ("maskImage,n", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("f1", bpo::value<bool>(&usePairF1),
       "Whether to use pair F1 other than traditional Rand index "
       "[default: true]")
      ("opt,g", bpo::value<int>(&globalOpt),
       "Global optimal assignment type (0: none, 1: RCC by merge, "
       "2: RCC by split) [default: 0]")
      ("optSplit,p", bpo::value<bool>(&optSplit),
       "Whether to determine based on optimal splits [default: false]")
      ("tweak,w", bpo::value<bool>(&tweak),
       "Whether to tweak conditions for thick boundaries [default: false]")
      ("mpd,d", bpo::value<double>(&maxPrecDrop),
       "Maximum precision drop allowed for merge [default: 1.0]")
      ("bclabel,l", bpo::value<std::string>(&bcLabelFile),
       "Output boundary label file name (optional)");
  return parse(argc, argv, opts) && operation()?
      EXIT_SUCCESS: EXIT_FAILURE;
}
