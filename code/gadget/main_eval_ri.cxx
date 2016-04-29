#include "type/big_num.hxx"
#include "util/stats.hxx"
#include "util/image_stats.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

bool operation (std::vector<std::string> const& resImageFiles,
                std::vector<std::string> const& refImageFiles,
                std::vector<std::string> const& maskImageFiles,
                bool adapted)
{
  int n = resImageFiles.size();
  std::vector<BigInt> TPs(n), TNs(n), FPs(n), FNs(n);
  // parfor(0, n, false, [&resImageFiles, &refImageFiles, &maskImageFiles,
  //                      &TPs, &TNs, &FPs, &FNs](int i){
  //          auto resImage =
  //            readImage<LabelImage<DIMENSION>>(resImageFiles[i]);
  //          auto refImage =
  //            readImage<LabelImage<DIMENSION>>(refImageFiles[i]);
  //          auto mask = maskImageFiles.size() <= i ||
  //            maskImageFiles[i].empty()?
  //            LabelImage<DIMENSION>::Pointer(nullptr):
  //            readImage<LabelImage<DIMENSION>>(maskImageFiles[i]);
  //          stats::pairStats(TPs[i], TNs[i], FPs[i], FNs[i], resImage,
  //                           refImage, mask, {}, {BG_VAL});
  //        }, 0);
  for (int i = 0; i < n; ++i) {
    auto resImage =
        readImage<LabelImage<DIMENSION>>(resImageFiles[i]);
    auto refImage =
        readImage<LabelImage<DIMENSION>>(refImageFiles[i]);
    auto mask = maskImageFiles.size() <= i ||
        maskImageFiles[i].empty()?
        LabelImage<DIMENSION>::Pointer(nullptr):
        readImage<LabelImage<DIMENSION>>(maskImageFiles[i]);
    stats::pairStats(TPs[i], TNs[i], FPs[i], FNs[i], resImage,
                     refImage, mask, {}, {BG_VAL});
  }
  BigInt TP = 0, TN = 0, FP = 0, FN = 0;
  for (int i = 0; i < n; ++i) {
    TP += TPs[i];
    TN += TNs[i];
    FP += FPs[i];
    FN += FNs[i];
  }
  if (adapted) { // Adapted Rand error
    double prec, rec, f;
    stats::precision(prec, TP, FP);
    stats::recall(rec, TP, FN);
    stats::f1(f, prec, rec);
    std::cout << prec << " " << rec << " " << 1.0 - f << std::endl;
  }
  else { // Traditional Rand error
    double ri;
    stats::randIndex(ri, TP, TN, FP, FN);
    std::cout << ri << std::endl;
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::vector<std::string> resImageFiles, refImageFiles, maskImageFiles;
  bool adapted = true;
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
       "Mask image file name(s) (optional)")
      ("adapted,a", bpo::value<bool>(&adapted),
       "Whether use adapted Rand error [default: true]");
  return
      parse(argc, argv, opts) &&
      operation(resImageFiles, refImageFiles, maskImageFiles, adapted)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
