#ifndef _glia_type_bignum_hxx_
#define _glia_type_bignum_hxx_

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "glia_base.hxx"

namespace glia {

typedef boost::multiprecision::int512_t BigInt;
typedef boost::multiprecision::cpp_dec_float_100 BigFloat;

};

#endif
