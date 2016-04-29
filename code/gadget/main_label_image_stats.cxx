#include "util/struct.hxx"
#include "util/image_io.hxx"
#include "util/stats.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& imageFile,
                std::string const& maskImageFile)
{
  auto image = readImage<LabelImage<DIMENSION>>(imageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  std::unordered_map<Label, unsigned int> cmap;
  genCountMap(cmap, image, mask);
  cmap.erase(BG_VAL);
  unsigned int smax = 0, smin = UINT_MAX;
  std::vector<unsigned int> sizes;
  sizes.reserve(cmap.size());
  for (auto const& cp: cmap) {
    smax = std::max(smax, cp.second);
    smin = std::min(smin, cp.second);
    sizes.push_back(cp.second);
  }
  unsigned int imageSize = 1;
  for (int i = 0; i < DIMENSION; ++i)
  { imageSize *= getImageSize(image, i); }
  std::vector<double> sh;
  stats::hist(sh, sizes, 20, std::make_pair(0.0, imageSize / 10.0));
  std::cout << "unique labels: " << cmap.size() << std::endl;
  std::cout << "min size: " << smin << std::endl;
  std::cout << "max size: " << smax << std::endl;
  std::cout << "size hist: ";
  for (auto x: sh) { std::cout << x << " "; }
  std::cout << std::endl;
  return true;
}


int main (int argc, char* argv[])
{
  std::string imageFile, maskImageFile;
  bpo::options_description opts("Usage:");
  opts.add_options()
      ("help", "Print usage info")
      ("image,i", bpo::value<std::string>(&imageFile)->required(),
       "Input image file name")
      ("mask,m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name (optional)");
  return parse(argc, argv, opts) && operation(imageFile, maskImageFile)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
