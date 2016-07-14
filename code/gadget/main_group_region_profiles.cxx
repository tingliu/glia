#include "util/struct.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
using namespace glia;

std::vector<std::string> inputSegImageFiles;
std::vector<std::string> maskImageFiles;
std::vector<int> inputImageIds;
std::vector<std::string> linkFiles;
bool relabel = false;
bool write16 = false;
bool compress = false;
std::vector<std::string> outputSegImageFiles;

bool operation ()
{
  typedef std::pair<int, Label> SRKey;
  typedef std::pair<SRKey, SRKey> Link;
  std::vector<Link> links;
  readData(links, linkFiles, true);
  int ns = inputSegImageFiles.size();
  std::vector<LabelImage<DIMENSION>::Pointer> segImages(ns);
  std::vector<LabelImage<DIMENSION>::Pointer> masks(ns);
  std::map<int, int> imap;
  for (int i = 0; i < ns; ++i) {
    segImages[i] = readImage<LabelImage<DIMENSION>>(inputSegImageFiles[i]);
    if (maskImageFiles.size() > i && maskImageFiles[i] != "NULL")
    { masks[i] = readImage<LabelImage<DIMENSION>>(maskImageFiles[i]); }
    imap[inputImageIds[i]] = i;
  }
  std::unordered_set<SRKey> regions;
  for (int i = 0; i < ns; ++i) {
    std::unordered_set<Label> keys;
    getKeys(keys, segImages[i], masks[i]);
    for (Label key : keys) {
      regions.insert(std::make_pair(inputImageIds[i], key));
    }
  }
  std::list<std::list<SRKey>> groups;
  groupRegions(groups, regions, links);
  std::vector<std::unordered_map<Label, Label>> lmaps(ns);
  Label groupLabel = 1;
  for (auto const& group : groups) {
    for (auto const& key : group)
    { lmaps[imap.find(key.first)->second][key.second] = groupLabel; }
    ++groupLabel;
  }
  parfor(0, ns, true, [&segImages, &masks, &lmaps](int i) {
      transformImage(segImages[i], lmaps[i], masks[i]); }, 0);
  if (outputSegImageFiles.size() == 1) {  // Write to one volume
    auto outputSegImage = stackImages(segImages);
    if (relabel) { relabelImage(outputSegImage, 0); }
    if (write16) {
      castWriteImage<UInt16Image<DIMENSION + 1>>(
          outputSegImageFiles.front(), outputSegImage, compress);
    } else {
      writeImage(outputSegImageFiles.front(), outputSegImage, compress);
    }
  } else {  // Write to slices
    if (relabel) { relabelImages(segImages, masks, 0); }
    for (int i = 0; i < ns; ++i) {
      if (write16) {
        castWriteImage<UInt16Image<DIMENSION>>(
            outputSegImageFiles[i], segImages[i], compress);
      } else {
        writeImage(outputSegImageFiles[i], segImages[i], compress);
      }
    }
  }
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("s", bpo::value<std::vector<std::string>>(
          &inputSegImageFiles)->required(),
       "Input segmentation image file names")
      ("m", bpo::value<std::vector<std::string>>(&maskImageFiles),
       "Input mask image file names (Use 'NULL' to bypass) (optional)")
      ("id", bpo::value<std::vector<int>>(&inputImageIds)->required(),
       "Input segmentation image ids")
      ("l", bpo::value<std::vector<std::string>>(&linkFiles)->required(),
       "Input link files")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output label image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("o", bpo::value<std::vector<std::string>>(
          &outputSegImageFiles)->required(), "Output segmentation image "
       "file name(s) (Use one/multiple file name(s) to save to "
       "one/slice-by-slice file(s)");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
