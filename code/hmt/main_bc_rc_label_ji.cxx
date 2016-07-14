#include "hmt/bc_label.hxx"
#include "hmt/tree_build.hxx"
#include "hmt/tree_greedy.hxx"
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

std::string rcLabelFile;
std::string bcLabelFile;
std::string segImageFile;
std::string mergeOrderFile;
std::string truthImageFile;
std::string maskImageFile;
std::vector<Label> pickedTruthRegionKeys;
double minJI = 0.5;

struct NodeData {
  Label label;
  double maxJI = -1.0;  // Max Jaccard index w.r.t. some truth region
  Label maxJIKey = BG_VAL;  // Corresponding truth region key
  int rcLabel = RC_LABEL_UNKNOWN;
  int bcLabel = BC_LABEL_UNKNOWN;
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
  RegionMap trmap(truthImage, mask, false);
  typedef TTree<NodeData> Tree;
  Tree tree;
  std::unordered_map<Label, int> leafKeyIndexMap;
  genTree(tree, order, [&leafKeyIndexMap](Tree::Node& node, Label r) {
      node.data.label = r;
      if (node.isLeaf()) { leafKeyIndexMap[r] = node.self; }
    });
  int nPickedTruthRegions = pickedTruthRegionKeys.size();
  std::vector<std::vector<std::pair<double, int>>>
      pickedTruthRegionJIs(nPickedTruthRegions);
  parfor(
      0, nPickedTruthRegions, true, [
          &rmap, &trmap, &segImage, &leafKeyIndexMap, &tree,
          &pickedTruthRegionJIs](int i) {
        auto const& tregion = trmap.find(pickedTruthRegionKeys[i])->second;
        int tarea = tregion.size();  // Truth region area
        std::unordered_map<Label, int> overlaps;  // Leaf overlaps
        stats::getOverlap(overlaps, tregion, segImage);
        std::unordered_set<int> overlapTreeNodes;  // All related tree nodes
        // Find all related tree nodes
        for (auto const& op : overlaps) {
          auto lit = leafKeyIndexMap.find(op.first);
          if (lit != leafKeyIndexMap.end()) {
            int leaf = lit->second;
            overlapTreeNodes.insert(leaf);
            tree.traverseAncestors(
                leaf, [&overlapTreeNodes](Tree::Node const& node) {
                  overlapTreeNodes.insert(node.self); });
          }
        }
        // Compute Jaccard index for each related tree node
        for (int ni : overlapTreeNodes) {
          int totalOverlap = 0;
          int totalArea = 0;
          tree.traverseLeaves(
              ni, [&totalOverlap, &totalArea, &overlaps, &rmap](
                  Tree::Node const& node) {
                auto oit = overlaps.find(node.data.label);
                if (oit != overlaps.end()) {
                  totalOverlap += oit->second;
                  totalArea += rmap.find(node.data.label)->second.size();
                }
              });
          double ji =
              (double)totalOverlap / (totalArea + tarea - totalOverlap);
          pickedTruthRegionJIs[i].push_back(std::make_pair(ji, ni));
        }
        std::sort(
            pickedTruthRegionJIs[i].begin(),
            pickedTruthRegionJIs[i].end(),
            std::greater<std::pair<double, int>>());
      }, 0);
  for (int ti = 0; ti < nPickedTruthRegions; ++ti) {
    for (auto const& jip : pickedTruthRegionJIs[ti]) {
      double ji = jip.first;
      int ni = jip.second;
      if (ji > tree[ni].data.maxJI) {
        tree[ni].data.maxJI = ji;
        tree[ni].data.maxJIKey = pickedTruthRegionKeys[ti];
      }
    }
  }
  // Greedily resolve tree w.r.t. max Jaccard index
  // Threshold picks w.r.t. Jaccard index for final rc/bc labels
  std::vector<int> picks;
  resolveTreeGreedy(
      picks, tree, [](Tree::Node const& node) {
        return node.data.maxJI >= minJI;
      }, [](Tree::Node const& lhs, Tree::Node const& rhs) {
        return lhs.data.maxJI < rhs.data.maxJI; });
  for (int ni : picks) {
    tree[ni].data.rcLabel = RC_LABEL_TRUE;
    tree[ni].data.bcLabel = BC_LABEL_MERGE;
    tree.traverseAncestors(ni, [](Tree::Node& node) {
        node.data.rcLabel = RC_LABEL_FALSE;
        node.data.bcLabel = BC_LABEL_SPLIT;
      });
    tree.traverseDescendants(ni, [](Tree::Node& node) {
        node.data.rcLabel = RC_LABEL_FALSE;
        node.data.bcLabel = BC_LABEL_MERGE;
      });
  }
  // Collect labels
  if (!rcLabelFile.empty()) {
    std::vector<int> rcLabels;
    rcLabels.reserve(tree.size());
    for (auto const& node : tree)
    { rcLabels.push_back(node.data.rcLabel); }
    writeData(rcLabelFile, rcLabels, "\n");
  }
  if (!bcLabelFile.empty()) {
    std::vector<int> bcLabels;
    bcLabels.reserve(order.size());
    for (auto const& node : tree)
    { if (!node.isLeaf()) { bcLabels.push_back(node.data.bcLabel); } }
    writeData(bcLabelFile, bcLabels, "\n");
  }
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
      ("tk", bpo::value<std::vector<Label>>(
          &pickedTruthRegionKeys)->multitoken(),
       "Picked truth region keys (e.g. --tk 1 2 5)")
      ("mji", bpo::value<double>(&minJI),
       "Minimum Jaccard index required [default: 0.5]")
      ("rclabel,r", bpo::value<std::string>(&rcLabelFile),
       "Output region label file name (optional)")
      ("bclabel,l", bpo::value<std::string>(&bcLabelFile),
       "Output boundary label file name (optional)");
  return parse(argc, argv, opts) && operation()?
      EXIT_SUCCESS: EXIT_FAILURE;
}
