#ifndef _glia_util_text_cmd_hxx_
#define _glia_util_text_cmd_hxx_

#include <boost/program_options.hpp>
#include "glia_base.hxx"

namespace glia {
namespace bpo = boost::program_options;

// Return true only if arguments parsing succeeds
inline bool parse (bpo::variables_map& vm, int argc, char** argv,
                   bpo::options_description& opts)
{
  try {
    bpo::store(bpo::parse_command_line(argc, argv, opts), vm);
    if (vm.count("help")) {
      std::cerr << opts << std::endl;
      return false;
    }
    bpo::notify(vm);
  }
  catch(bpo::error& e) {
    if (!vm.empty())
    { std::cerr << "Error: " << e.what() << std::endl; }
    std::cerr << opts << std::endl;
    return false;
  }
  return true;
}


// Return true only if arguments parsing succeeds
inline bool parse (int argc, char** argv, bpo::options_description& opts)
{
  bpo::variables_map vm;
  return parse(vm, argc, argv, opts);
}

};

#endif
