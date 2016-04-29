#ifndef _glia_type_tuple_hxx_
#define _glia_type_tuple_hxx_

#include "type/object.hxx"

namespace glia {

template <typename TX0, typename TX1 = TX0, typename TX2 = TX1>
class TTriple : public Object {
 public:
  typedef TTriple<TX0, TX1, TX2> Self;

  TX0 x0;
  TX1 x1;
  TX2 x2;

  TTriple () {}

  ~TTriple () override {}

  TTriple (TX0 const& x0, TX1 const& x1, TX2 const& x2)
      : x0(x0), x1(x1), x2(x2) {}

  friend std::ostream& operator<< (std::ostream& os, Self const& t)
  { return os << t.x0 << " " << t.x1 << " " << t.x2; }

  friend std::istream& operator>> (std::istream& is, Self& t)
  { return is >> t.x0 >> t.x1 >> t.x2; }
};


template <typename TX0, typename TX1 = TX0, typename TX2 = TX1,
          typename TX3 = TX2>
class TQuad : public TTriple<TX0, TX1, TX2> {
 public:
  typedef TTriple<TX0, TX1, TX2> Super;
  typedef TQuad<TX0, TX1, TX2, TX3> Self;

  TX3 x3;

  TQuad () {}

  ~TQuad () override {}

  TQuad (TX0 const& x0, TX1 const& x1, TX2 const& x2, TX3 const& x3)
      : Super(x0, x1, x2), x3(x3) {}

  friend std::ostream& operator<< (std::ostream& os, Self const& q)
  { return os << static_cast<Super const&>(q) << " " << q.x3; }

  friend std::istream& operator>> (std::istream& is, Self& q)
  { return is >> static_cast<Super&>(q) >> q.x3; }
};

};

#endif
