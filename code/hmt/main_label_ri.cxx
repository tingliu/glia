#include "hmt/bc_label.hxx"
#include "hmt/tree_build.hxx"
#include "type/big_num.hxx"
#include "type/tuple.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

struct NodeData {
  Label label;
  int bcLabel = BC_LABEL_MERGE;
};

bool operation (std::string const& bcLabelFile,
                std::string const& rcLabelFile,
                std::string const& segImageFile,
                std::string const& mergeOrderFile,
                std::string const& truthImageFile,
                std::string const& maskImageFile,
                bool tweak, double maxPrecDrop, double minJIndex)
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
  std::vector<int> bcLabels;
  int m = order.size();
  bcLabels.resize(m);
  parfor(0, m, true, [&bcLabels, &rmap, &order, &truthImage, tweak,
                      maxPrecDrop](int i){
           double mergeF1, splitF1;
           bcLabels[i] = genBoundaryClassificationLabelF1<BigInt>
               (mergeF1, splitF1, rmap, order[i].x0, order[i].x1,
                order[i].x2, truthImage, tweak, maxPrecDrop);
         }, 0);
  if (!bcLabelFile.empty()) { writeData(bcLabelFile, bcLabels, "\n"); }
  if (!rcLabelFile.empty()) {
    std::unordered_map<Label, unsigned int> tcmap;
    genCountMap(tcmap, truthImage, mask);
    tcmap.erase(BG_VAL);
    typedef TTree<NodeData> Tree;
    Tree tree;
    auto bit = bcLabels.begin();
    genTree(tree, order, [&bit](Tree::Node& node, Label r) {
        node.data.label = r;
        if (!node.isLeaf()) { node.data.bcLabel = *bit++; }
      });
     int n = tree.size();
     std::vector<int> rcLabels(n, RC_LABEL_FALSE);
     parfor(0, n, true, [&rmap, &tree, &rcLabels, &truthImage, &tcmap,
                         minJIndex](int i) {
              auto const& tn = tree[i];
              if (tn.data.bcLabel == BC_LABEL_MERGE && tn.parent >= 0 &&
                  tree[tn.parent].data.bcLabel == BC_LABEL_SPLIT) {
                std::unordered_map<Label, unsigned int> cmap;
                auto const& region = rmap.find(tn.data.label)->second;
                region.traverse(
                    [&cmap, &truthImage](
                        RegionMap::Region::Point const& p) {
                      auto val = truthImage->GetPixel(p);
                      if (val != BG_VAL) {
                        ++citerator(cmap, val, 0)->second;
                      }
                    });
                for (auto const& cp: cmap) {
                  double ji = sdivide(cp.second, region.size() +
                                      tcmap.find(cp.first)->second -
                                      cp.second, 0.0);
                  if (ji >= minJIndex) {
                    rcLabels[i] = RC_LABEL_TRUE;
                    break;
                  }
                }
              }
            }, 0);
     writeData(rcLabelFile, rcLabels, "\n");
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::string bcLabelFile, rcLabelFile, mergeOrderFile, segImageFile,
      truthImageFile, maskImageFile;
  bool tweak = false;
  double maxPrecDrop = 1.0, minJIndex = 0.0;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("truthImage,t", bpo::value<std::string>(&truthImageFile)->required(),
       "Input ground truth segmentation image file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("tweak,w", bpo::value<bool>(&tweak),
       "Whether to tweak conditions for thick boundaries [default: false]")
      ("mpd,d", bpo::value<double>(&maxPrecDrop),
       "Maximum precision drop allowed for merge [default: 1.0]")
      ("mji,j", bpo::value<double>(&minJIndex),
       "Minimum Jaccard index allowed for region [default: 0.0]")
      ("bclabel,l", bpo::value<std::string>(&bcLabelFile),
       "Output boundary label file name (optional)")
      ("rclabel,r", bpo::value<std::string>(&rcLabelFile),
       "Output region label file name (optional)");
  return
      parse(argc, argv, opts) &&
      operation(bcLabelFile, rcLabelFile, segImageFile, mergeOrderFile,
                truthImageFile, maskImageFile, tweak, maxPrecDrop,
                minJIndex)? EXIT_SUCCESS: EXIT_FAILURE;
}
