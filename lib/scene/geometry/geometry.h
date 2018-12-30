#pragma once

#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_utils.h"
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

    using CudaGeometryCache = std::unordered_map<Geometry*, CudaGeometry*>;
    virtual CudaGeometry* create_cuda(CudaGeometryCache& cache) const = 0;

private:
    std::shared_ptr<CudaGeometry> _cached_cuda;
};

using GeometryPtr = std::shared_ptr<Geometry>;

}
