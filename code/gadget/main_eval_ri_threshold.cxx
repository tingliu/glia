#include "type/big_num.hxx"
#include "type/tuple.hxx"
#include "util/container.hxx"
#include "util/stats.hxx"
#include "util/image_stats.hxx"
#include "util/image_io.hxx"
#include "util/image_alg.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

std::vector<std::string> resImageFiles;
std::vector<std::string> refImageFiles;
std::vector<std::string> maskImageFiles;
double lower = 0.0;
double upper = 1.0;
int nThreshold = 10;
bool adapted = true;
bool useWatershed = false;

bool operation ()
{
  int n = resImageFiles.size();
  double step = (upper - lower) / nThreshold;
  std::vector<double> thresholds;
  crange(thresholds, lower, step, upper);
  // (TP, TN, FP, FN)
  std::vector<TQuad<BigInt>> scores(nThreshold, TQuad<BigInt>(0, 0, 0, 0));
  for (int i = 0; i < n; ++i) {
    auto refImage = readImage<LabelImage<DIMENSION>>(refImageFiles[i]);
    auto resImage = readImage<RealImage<DIMENSION>>(resImageFiles[i]);
    auto mask = maskImageFiles.size() <= i || maskImageFiles[i].empty()?
        LabelImage<DIMENSION>::Pointer(nullptr):
        readImage<LabelImage<DIMENSION>>(maskImageFiles[i]);
    for (int j = 0; j < nThreshold; ++j) {
      auto canvas = useWatershed ?
          watershed<LabelImage<DIMENSION>>(resImage, thresholds[j]):
          thresholdImage<LabelImage<DIMENSION>>(
              resImage, lower, thresholds[j], 1, 0);
      canvas = labelConnectedComponents<LabelImage<DIMENSION>>(canvas);
      BigInt TP, TN, FP, FN;
      stats::pairStats(
          TP, TN, FP, FN, canvas, refImage, mask, {}, {BG_VAL});
      scores[j].x0 += TP;
      scores[j].x1 += TN;
      scores[j].x2 += FP;
      scores[j].x3 += FN;
    }
  }
  for (int j = 0; j < nThreshold; ++j) {
    if (adapted) {
      double prec, rec, f;
      stats::precision(prec, scores[j].x0, scores[j].x2);
      stats::recall(rec, scores[j].x0, scores[j].x3);
      stats::f1(f, prec, rec);
      std::cout << thresholds[j] << " " << prec << " " << rec << " "
                << 1.0 - f << std::endl;
    } else {
      double ri;
      stats::randIndex(
          ri, scores[j].x0, scores[j].x1, scores[j].x2, scores[j].x3);
      std::cout << thresholds[j] << " " << ri << std::endl;
    }
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("resImage,p",
       bpo::value<std::vector<std::string>>(&resImageFiles)->required(),
       "Input proposed image file name(s)")
      ("refImage,r",
       bpo::value<std::vector<std::string>>(&refImageFiles)->required(),
       "Input reference image file name(s)")
      ("mask,m",
       bpo::value<std::vector<std::string>>(&maskImageFiles),
       "Mask image file name(s)")
      ("lower,l", bpo::value<double>(&lower),
       "Lower bound threshold [default: 0.0]")
      ("upper,u", bpo::value<double>(&upper),
       "Upper bound threshold [default: 1.0]")
      ("nThreshold,t", bpo::value<int>(&nThreshold),
       "Number of thresholds [default: 10]")
      ("adapted,a", bpo::value<bool>(&adapted),
       "Whether use adapted Rand error [default: true]")
      ("watershed,w", bpo::value<bool>(&useWatershed),
       "Whether use watershed instead of thresholding [default: false]");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS: EXIT_FAILURE;
}
