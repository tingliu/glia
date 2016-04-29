#include "hmt/tree_build.hxx"
#include "hmt/tree_segment.hxx"
#include "hmt/tree_ccm.hxx"
#include "type/tuple.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;
using namespace glia::hmt;

struct NodeData {
  Label label;
  double Em, Es;
};

std::string mergeOrderFile;
std::string mergeProbFile;
std::string segImageFile;
std::string maskImageFile;
bool ignore = true;
bool write16 = false;
bool relabel = false;
bool compress = false;
std::string bcImageFile;
std::string finalSegImageFile;


bool operation ()
{
  typedef TTree<NodeData> Tree;
  std::vector<TTriple<Label>> order;
  readData(order, mergeOrderFile, true);
  std::vector<double> mergeProbs;
  readData(mergeProbs, mergeProbFile, true);
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  Tree tree;
  auto mpit = mergeProbs.begin();
  genTree(tree, order, [&mpit](Tree::Node& node, Label r) {
      node.data.label = r;
      if (node.isLeaf()) {
        node.data.Em = 0.0;
        node.data.Es = FMAX;
      }
      else {
        double p = *mpit++;
        node.data.Em = isfeq(p, 0.0)? FMAX: -std::log(p);
        p = 1.0 - p;
        node.data.Es = isfeq(p, 0.0)? FMAX: -std::log(p);
      }
    });
  std::vector<std::pair<double, double>> Ems; // Energy tuples
  computeEnergyTuples(Ems, tree);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  // Generate and output boundary confidence image
  if (!bcImageFile.empty()) {
    auto bcImage = createImage<RealImage<DIMENSION>>
        (segImage->GetRequestedRegion(), 0.0);
    TRegionMap<Label, Point<DIMENSION>> rmap(segImage, mask, order, true);
    // std::vector<std::pair<double, std::vector<double>>> Ems;
    // computeFactorTreeFullLabelEnergy(Ems, tree);
    // double normConst = 0.0;
    // for (auto x: Ems[tree.root()].second) { normConst += std::exp(-x); }
    // genBoundaryConfidenceImage
    //   (bcImage, rmap, tree, std::vector<int>(), [normConst, &Ems, &tree]
    //    (Tree::Node const& node) -> double
    //    {
    //      std::vector<double> Emargin;
    //      computeFactorNodeMarginalEnergy(Emargin, tree, node.self, Ems);
    //      double ret = 0.0;
    //      for (auto x: Emargin) { ret += std::exp(-x); }
    //      return ret / normConst;
    //    });
    genBoundaryConfidenceImage
        (bcImage, rmap, tree, std::vector<int>(), [&Ems, &tree]
         (Tree::Node const& node) -> double
         {
           double
               posEnergy =
               computeFactorNodeEnergyPositive(tree, node.self, Ems),
               negEnergy =
               computeFactorNodeEnergyNegative(tree, node.self, Ems);
           return posEnergy / stats::plusEqual(negEnergy, posEnergy);
         });
    writeImage(bcImageFile, bcImage, compress);
  }
  // Resolve factor tree and output final segmentation
  if (!finalSegImageFile.empty()) {
    std::vector<int> picks;
    resolveFactorTree(picks, tree, Ems);
    genFinalSegmentation(segImage, tree, picks, mask, 1u, !ignore);
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
      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
       "Input merging order file name")
      ("mergeProbs,p", bpo::value<std::string>(&mergeProbFile)->required(),
       "Input merging probability file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("ignore,i", bpo::value<bool>(&ignore),
       "Ignore missing regions by labeling them BG_VAL [default: true]")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output label image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("finalSegImage,f",
       bpo::value<std::string>(&finalSegImageFile),
       "Output final segmentation image file name (optional)")
      ("bcImage,b", bpo::value<std::string>(&bcImageFile),
       "Output boundary confidence image file name (optional)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
