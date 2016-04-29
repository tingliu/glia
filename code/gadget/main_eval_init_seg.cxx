#include "type/big_num.hxx"
#include "type/region_map.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/image_stats.hxx"
#include "util/mp.hxx"
using namespace glia;

bool operation (std::string const& outputSegImageFile,
                std::string const& segImageFile,
                std::string const& maskImageFile,
                std::string const& truthImageFile,
                bool relabel, bool write16, bool compress)
{
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  auto segImage = readImage<LabelImage<DIMENSION>>(segImageFile);
  auto truthImage = readImage<LabelImage<DIMENSION>>(truthImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  RegionMap rmap(segImage, mask, false);
  int rn = rmap.size();
  std::vector<std::pair<RegionMap::const_iterator, Label>> rtmap;
  rtmap.reserve(rn);
  for (auto rit = rmap.begin(); rit != rmap.end(); ++rit)
  { rtmap.emplace_back(rit, BG_VAL); }
  parfor(0, rn, false, [&rtmap, &truthImage](int i){
      std::unordered_map<Label, unsigned int> cmap;
      rtmap[i].first->second.traverse
          ([&truthImage, &cmap](RegionMap::Region::Point const& p)
           { ++citerator(cmap, truthImage->GetPixel(p), 0)->second; });
      cmap.erase(BG_VAL); // Ignore background pixels
      unsigned int maxCnt = 0;
      Label maxKey = BG_VAL;
      for (auto const& cp: cmap) {
        if (cp.second > maxCnt) {
          maxKey = cp.first;
          maxCnt = cp.second;
        }
      }
      rtmap[i].second = maxKey;
    }, 0);
  std::unordered_map<Label, Label> lmap; // seg -> truth
  for (auto const& rtp: rtmap) { lmap[rtp.first->first] = rtp.second; }
  transformImage(segImage, lmap, mask, true);
  double f, prec, rec;
  stats::pairF1<BigInt>(f, prec, rec, segImage, truthImage, mask, {},
    {BG_VAL});
  std::cout << prec << " " << rec << " " << 1.0 - f << std::endl;
  if (!outputSegImageFile.empty()) {
    if (relabel) { relabelImage(segImage, 0); }
    if (write16) {
      castWriteImage<UInt16Image<DIMENSION>>(outputSegImageFile, segImage,
                                             compress);
    }
    else { writeImage(outputSegImageFile, segImage, compress); }
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::string segImageFile, maskImageFile, truthImageFile,
      outputSegImageFile;
  bool relabel = false, write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
       "Input initial segmentation image file name")
      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
       "Input mask image file name (optional)")
      ("truthImage,t", bpo::value<std::string>(&truthImageFile)->required(),
       "Input truth image file name")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output label image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image file [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,f", bpo::value<std::string>(&outputSegImageFile),
       "Output final segmentation image file (optional)");
  return
      parse(argc, argv, opts) &&
      operation(outputSegImageFile, segImageFile, maskImageFile,
                truthImageFile, relabel, write16, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
