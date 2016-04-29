#include "type/tree.hxx"
#include "hmt/bc_label.hxx"
#include "hmt/tree_build.hxx"
#include "type/tuple.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

std::string segImageFile;
std::string mergeOrderFile;
std::vector<std::string> truthImageFiles;
std::string maskImageFile;
int globalOpt = 0;  // 0: Bypass, 1: RCC by merge, 2: RCC by split
std::string labelFile;

struct NodeData {
  Label label;
  std::vector<int> bestSplits;
  int bcLabel;
};

bool operation ()
{
  std::vector<TTriple<Label>> order;
  readData(order, mergeOrderFile, true);
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  int nt = truthImageFiles.size();
  std::vector<LabelImage<DIMENSION>::Pointer> truthImages(nt);
  for (int i = 0; i < nt; ++i) {
    truthImages[i] = readImage<LabelImage<DIMENSION>>(truthImageFiles[i]);
  }
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, order, false);
  std::vector<int> bcLabels;
  int n = order.size();
  if (globalOpt == 0) {
    // Local optimal
    bcLabels.resize(n);
    parfor(0, n, true, [&bcLabels, &rmap, &order, &truthImages](int i){
        std::vector<int> tmpLabels(truthImages.size());
        for (int l = 0; l < truthImages.size(); ++l) {
          double mergeVI, splitVI;
          tmpLabels[l] = genBoundaryClassificationLabelVI(
              mergeVI, splitVI, rmap, order[i].x0, order[i].x1,
              order[i].x2, truthImages[l]);
        }
        bcLabels[i] = majority(tmpLabels);
      }, 0);
  } else {
    // Global optimal
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
        std::vector<int> tmpLabels(nt);
        for (int i = 0; i < nt; ++i) {
          double mergeVI = stats::vi(
              pMergeRegions, truthImages[i], {BG_VAL});
          double splitVI = stats::vi(
              pSplitRegions, truthImages[i], {BG_VAL});
          tmpLabels[i] =
              mergeVI < splitVI ? BC_LABEL_MERGE : BC_LABEL_SPLIT;
        }
        if (majority(tmpLabels) == BC_LABEL_MERGE) {
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
              node.data.bcLabel = BC_LABEL_MERGE;
            });
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
  writeData(labelFile, bcLabels, "\n");
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s",
       bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("mergeOrder,o",
       bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("truthImage,t",
       bpo::value<std::vector<std::string>>(&truthImageFiles)->required(),
       "Input ground truth segmentation image file name(s)")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("opt,g", bpo::value<int>(&globalOpt),
       "Global optimal assignment type (0: none, 1: RCC by merge, "
       "2: RCC by split) [default: 0]")
      ("bcLabels,l", bpo::value<std::string>(&labelFile)->required(),
       "Output label file name");
  return parse(argc, argv, opts) && operation()?
      EXIT_SUCCESS: EXIT_FAILURE;
}
