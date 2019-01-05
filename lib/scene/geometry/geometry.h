#pragma once

#include "gpgpu/gpgpu_converter.h"
#include "scene/object.h"
#include "scene/geometry/vertex_container.h"
#include <unordered_map>

namespace cpt {

class Geometry: public Object
{
public:
    enum class PrimitiveType
    {
        None,
        Triangle
    };

    virtual bool is_primitive() const { return false; }
    virtual PrimitiveType primitive_type() const { return PrimitiveType::None; }

};

using GeometryPtr = std::shared_ptr<Geometry>;

}
