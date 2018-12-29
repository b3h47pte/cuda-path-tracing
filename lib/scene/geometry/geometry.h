#pragma once

#include "gpgpu/cuda_utils.h"
#include "scene/object.h"
#include "scene/geometry/vertex_container.h"

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
