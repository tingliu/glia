#include "util/image_stats.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

bool operation (std::vector<std::string> const& resImageFiles,
                std::vector<std::string> const& refImageFiles,
                std::vector<std::string> const& maskImageFiles)
{
  int n = resImageFiles.size();
  std::vector<double> fss(n), fms(n);
  for (int i = 0; i < n; ++i) {}
  parfor(0, n, false, [&resImageFiles, &refImageFiles, &maskImageFiles,
                       &fss, &fms](int i){
           auto resImage =
               readImage<LabelImage<DIMENSION>>(resImageFiles[i]);
           auto refImage =
               readImage<LabelImage<DIMENSION>>(refImageFiles[i]);
           auto mask = maskImageFiles.empty() || maskImageFiles[i].empty()?
               LabelImage<DIMENSION>::Pointer(nullptr):
               readImage<LabelImage<DIMENSION>>(maskImageFiles[i]);
           fss[i] = stats::centropy(refImage, resImage, mask, {BG_VAL}, {});
           fms[i] = stats::centropy(resImage, refImage, mask, {}, {BG_VAL});
         }, 0);
  double fs = stats::mean(fss), fm = stats::mean(fms); // false split/merge
  std::cout << fs << " " << fm << " " << fs + fm << std::endl;
  return true;
}


int main (int argc, char* argv[])
{
  std::vector<std::string> resImageFiles, refImageFiles, maskImageFiles;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("resImage,p",
       bpo::value<std::vector<std::string>>(&resImageFiles)->required(),
       "Input proposed image file name(s)")
      ("refImage,r",
       bpo::value<std::vector<std::string>>(&refImageFiles)->required(),
       "Input reference image file name(s)")
      ("mask,m", bpo::value<std::vector<std::string>>(&maskImageFiles),
       "Mask image file name(s)");
  return
      parse(argc, argv, opts) &&
      operation(resImageFiles, refImageFiles, maskImageFiles)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
