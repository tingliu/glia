#ifndef _glia_type_object_hxx_
#define _glia_type_object_hxx_

#include "glia_base.hxx"

namespace glia {

class Object {
 public:
  typedef Object Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  virtual ~Object () = 0;
};

inline Object::~Object () {}

};

#endif
