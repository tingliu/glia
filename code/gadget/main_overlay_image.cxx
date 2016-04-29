#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::string labelImageFile;
std::string bgImageFile;
std::string maskImageFile;
double opacity = 0.6;
bool drawBoundary = true;
std::vector<int> boundaryRGB{BG_VAL, BG_VAL, BG_VAL};
std::string outputImageFile;

bool operation ()
{
  auto labelImage = readImage<LabelImage<DIMENSION>>(labelImageFile);
  auto bgImage = bgImageFile.empty()?
      UInt8Image<DIMENSION>::Pointer(nullptr):
      readImage<UInt8Image<DIMENSION>>(bgImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(labelImageFile);
  auto outputImage = overlayImage(labelImage, bgImage, mask, opacity);
  if (drawBoundary) {
    Rgb bgVal;
    bgVal.Set(boundaryRGB[0], boundaryRGB[1], boundaryRGB[2]);
    std::vector<Point<DIMENSION>> bgPoints;
    getPoints(bgPoints, labelImage, mask, BG_VAL, 0);
    if (!bgPoints.empty()) {
      for (auto const & p : bgPoints) { outputImage->SetPixel(p, bgVal); }
    } else {
      typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
      RegionMap rmap(labelImage, mask, true);
      for (auto const& rp: rmap) {
        rp.second.boundary.traverse
            ([&outputImage, &bgVal](RegionMap::Region::Point const& p)
             { outputImage->SetPixel(p, bgVal); });
      }
    }
  }
  writeImage(outputImageFile, outputImage, false);
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("labelImage,l", bpo::value<std::string>(&labelImageFile)->required(),
       "Input label image file name")
      ("bgImage,i", bpo::value<std::string>(&bgImageFile),
       "Input background (uint8) image file name (optional)")
      ("mask,m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name (optional)")
      ("opacity,p", bpo::value<double>(&opacity), "Opacity [default: 0.6]")
      ("drawb,b", bpo::value<bool>(&drawBoundary),
       "Whether to draw boundary pixels [default: true]")
      ("bc", bpo::value<std::vector<int>>(&boundaryRGB)->multitoken(),
       "Boundary RGB [default: 0 0 0]")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return parse(argc, argv, opts) && operation()?
      EXIT_SUCCESS: EXIT_FAILURE;
}
