#include "util/struct_merge.hxx"
#include "util/container.hxx"
#include "util/image_io.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool comp (TTriple<Label> const& m0, TTriple<Label> const& m1)
{ return m0.x2 < m1.x2; }


bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                std::string const& maskImageFile,
                std::vector<std::string> const& mergeOrderFiles,
                bool relabel, bool write16, bool compress)
{
  int n = mergeOrderFiles.size();
  std::vector<std::vector<TTriple<Label>>> tmerges(n);
  for (int i = 0; i < n; ++i)
  { readData(tmerges[i], mergeOrderFiles[i], true); }
  std::vector<TTriple<Label>> merges;
  merges.reserve(count(merges, [](std::vector<TTriple<Label>> const& m)
                       -> unsigned int { return m.size(); }));
  for (int i = 0; i < n; ++i) { splice(merges, tmerges[i]); }
  std::sort(merges.begin(), merges.end(), comp);
  std::unordered_map<Label, Label> lmap;
  transformKeys(lmap, merges);
  auto image = readImage<LabelImage<DIMENSION>>(inputImageFile);
  auto mask = maskImageFile.empty()?
      LabelImage<DIMENSION>::Pointer(nullptr):
      readImage<LabelImage<DIMENSION>>(maskImageFile);
  transformImage(image, lmap, mask, false);
  if (relabel) { relabelImage(image, 0); }
  if (write16) {
    castWriteImage<UInt16Image<DIMENSION>>
        (outputImageFile, image, compress);
  }
  else { writeImage(outputImageFile, image, compress); }
  return true;
}


int main (int argc, char* argv[])
{
  std::string outputImageFile, inputImageFile, maskImageFile;
  std::vector<std::string> mergeOrderFiles;
  bool relabel = false, write16 = false, compress = false;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input image file name")
      ("mask,m", bpo::value<std::string>(&maskImageFile),
       "Mask image file name")
      ("merge,g",
       bpo::value<std::vector<std::string>>(&mergeOrderFiles)->required(),
       "Input merge order file name(s)")
      ("relabel,r", bpo::value<bool>(&relabel),
       "Whether to relabel output image [default: false]")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write to uint16 image [default: false]")
      ("compress,z", bpo::value<bool>(&compress),
       "Whether to compress output image file(s) [default: false]")
      ("outputImage,o", bpo::value<std::string>(&outputImageFile)->required(),
       "Output image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, maskImageFile,
                mergeOrderFiles, relabel, write16, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
