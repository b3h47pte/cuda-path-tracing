#pragma once

#include "math/transform.h"
#include <memory>

namespace cpt {

class GpgpuConverter;

class Object
{
public:
    virtual ~Object() {}

    void set_object_to_world_xform(const Transform& xform) { _object_to_world_xform = xform; }
    const Transform& object_to_world_xform() const { return _object_to_world_xform; }

    virtual void convert(GpgpuConverter& converter) const {}
private:
    Transform _object_to_world_xform;
};

using ObjectPtr = std::shared_ptr<Object>;

}
