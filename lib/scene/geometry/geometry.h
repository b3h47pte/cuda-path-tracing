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

    virtual void convert(GpgpuConverter& converter) const = 0;
};

using GeometryPtr = std::shared_ptr<Geometry>;

}
