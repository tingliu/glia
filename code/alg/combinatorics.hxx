#ifndef _glia_alg_combinatorics_hxx_
#define _glia_alg_combinatorics_hxx_

#include "glia_base.hxx"

namespace glia {
namespace alg {

template <typename TContainerOut, typename TContainerIn,
          typename Func> void
combination
(TContainerOut& ret, typename TContainerOut::value_type res,
 std::vector<TContainerIn const*> const& pcs,
 typename std::vector<TContainerIn const*>::const_iterator pcit,
 Func f)
{
  if (pcit == pcs.end()) {
    ret.insert(ret.end(), res);
    return;
  }
  auto nit = std::next(pcit);
  for (auto const& x: **pcit) {
    auto tmpRes = res;
    f(tmpRes, x);
    combination(ret, tmpRes, pcs, nit, f);
  }
}

};
};

#endif
