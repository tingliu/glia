#include "util/struct.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

std::vector<std::string> regionPairFiles;
std::vector<std::string> pairScoreFiles;
double minScore;
bool forceLink = true;
std::string linkFile;

bool operation ()
{
  typedef std::pair<int, Label> SRKey;
  typedef std::pair<SRKey, SRKey> Link;
  std::vector<std::pair<SRKey, SRKey>> regionPairs;
  readData(regionPairs, regionPairFiles, true);
  std::vector<double> pairScores;
  readData(pairScores, pairScoreFiles, true);
  std::unordered_map<SRKey, std::priority_queue<std::pair<double, Link>>>
      weakRegionLinkMap;
  std::unordered_set<SRKey> regions;
  std::vector<Link> links;
  int np = regionPairs.size();
  for (int i = 0; i < np; ++i) {
    auto const& link = regionPairs[i];
    regions.insert(link.first);
    regions.insert(link.second);
    double score = pairScores[i];
    if (score >= minScore) { links.push_back(link); }
    else if (forceLink) {
      weakRegionLinkMap[link.first].push(
          std::make_pair(pairScores[i], link));
      weakRegionLinkMap[link.second].push(
          std::make_pair(pairScores[i], link));
    }
  }
  if (forceLink) {
    std::list<std::list<SRKey>> groups;
    groupRegions(groups, regions, links);
    for (auto const& group : groups) {
      if (group.size() == 1) {
        links.push_back(
            weakRegionLinkMap.find(group.front())->second.top().second);
      }
    }
  }
  writeData(linkFile, links, "\n");
  return true;
}


int main (int argc, char* argv[])
{
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("p", bpo::value<std::vector<std::string>>(
          &regionPairFiles)->required(),
       "Input region pair file name(s)")
      ("s", bpo::value<std::vector<std::string>>(
          &pairScoreFiles)->required(),
       "Input region pair score file name(s)")
      ("ms", bpo::value<double>(&minScore)->required(),
       "Min pair score to link")
      ("fl", bpo::value<bool>(&forceLink),
       "Whether to force at least one link [default: true]")
      ("l", bpo::value<std::string>(&linkFile)->required(),
       "Output link file name");
  return parse(argc, argv, opts) && operation() ?
      EXIT_SUCCESS : EXIT_FAILURE;
}
