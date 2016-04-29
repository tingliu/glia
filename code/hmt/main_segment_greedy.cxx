#include "hmt/tree_build.hxx"
#include "hmt/tree_segment.hxx"
#include "hmt/tree_greedy.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;
using namespace glia::hmt;

std::string segImageFile;
std::vector<std::string> mergeOrderFiles;
std::vector<std::string> mergeProbFiles;
std::vector<std::string> regionProbFiles;
std::string maskImageFile;
bool ignore = true;
bool relabel = false;
bool write16 = false;
bool compress = false;
std::string finalSegImageFile;
std::string bcImageFile;

struct NodeData {
  Label label;
  double potential;
};


bool operation ()
{
  typedef TTree<NodeData> Tree;
  int nTree = mergeOrderFiles.size();
  std::vector<std::vector<TTriple<Label>>> orders(nTree);
  std::vector<Tree> trees(nTree);
  parfor(0, nTree, false, [&orders, &trees](int i) {
      readData(orders[i], mergeOrderFiles[i], true);
      if (mergeProbFiles.size() > i) {
        std::vector<double> mergeProbs;
        readData(mergeProbs, mergeProbFiles[i], true);
        // Generate tree and initialize node potentials
        // For root and leaves, use squared probabilities as potentials
        genTreeWithNodePotentials
            (trees[i], orders[i], mergeProbs.begin());
      } else {
        genTree(trees[i], orders[i], [](Tree::Node& node, Label r) {
            node.data.label = r;
            node.data.potential = 1.0;
          });
      }
      // Update with unary (region) potential if available
      if (!regionProbFiles.empty()) {
        std::vector<double> regionProbs;
        readData(regionProbs, regionProbFiles[i], true);
        auto rpit = regionProbs.begin();
        for (auto& tn: trees[i])
        { tn.data.potential *= std::max(*rpit++, FEPS); }
      }
    }, 0);
  // Load images
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  // Generate and output boundary confidence image
  if (!bcImageFile.empty()) {
    auto bcImage = createImage<RealImage<DIMENSION>>
        (segImage->GetRequestedRegion(), 0.0);
    TRegionMap<Label, Point<DIMENSION>> rmap(segImage, mask, true);
    std::vector<TRegionMap<Label, Point<DIMENSION>>> rmaps(nTree, rmap);
    for (int i = 0; i < nTree; ++i) { rmaps[i].set(orders[i]); }
    genBoundaryConfidenceImage
        (bcImage, rmaps, trees, std::vector<std::pair<int, int>>(),
         [](Tree::Node const& node) -> Real
         { return node.data.potential; });
    writeImage(bcImageFile, bcImage, compress);
  }
  // Resolve tree and output final segmentation
  if (!finalSegImageFile.empty()) {
    // Resolve tree
    std::vector<std::pair<int, int>> picks;
    resolveTreeGreedy
        (picks, trees, [](Tree::Node const& node0, Tree::Node const& node1)
         -> bool { return node0.data.potential < node1.data.potential; });
    // Output final segmentation
    genFinalSegmentation(segImage, trees, picks, mask, 1u, !ignore);
    if (relabel) { relabelImage(segImage, 0); }
    if (write16) {
      castWriteImage<UInt16Image<DIMENSION>>
          (finalSegImageFile, segImage, compress);
    }
    else { writeImage(finalSegImageFile, segImage, compress); }
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
      ("mergeOrders,o",
       bpo::value<std::vector<std::string>>(&mergeOrderFiles)->required(),
       "Input merging order file name(s)")
      ("mergeProbs,p",
       bpo::value<std::vector<std::string>>(&mergeProbFiles),
       "Input merging probability file name(s)")
      ("regionProbs,n",
       bpo::value<std::vector<std::string>>(&regionProbFiles),
       "Input region probability file name(s)")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("ignore,i", bpo::value<bool>(&ignore),
       "Ignore missing regions by labeling them BG_VAL [default: true]")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output label image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("finalSegImage,f",
       bpo::value<std::string>(&finalSegImageFile),
       "Output final segmentation image file name (optional)")
      ("bcImage,b", bpo::value<std::string>(&bcImageFile),
       "Output boundary confidence image file name (optional)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
