#include "util/image_io.hxx"
#include "util/struct.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string segImageFile;
std::string maskImageFile;
bool includeBG = false;

bool operation ()
{
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  std::map<Label, int> cmap;
  genCountMap(cmap, segImage, mask);
  if (!includeBG) { cmap.erase(BG_VAL); }
  for (auto const& cp : cmap)
  { std::cout << cp.first << " " << cp.second << std::endl; }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("includeBG,b", bpo::value<bool>(&includeBG),
       "Whether to include background [default: false]");
  return parse(argc, argv, opts) && operation()?
      EXIT_SUCCESS: EXIT_FAILURE;
}
