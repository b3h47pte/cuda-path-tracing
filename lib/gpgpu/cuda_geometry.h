#pragma once

#include "gpgpu/cuda_aabb.h"
#include <vector>

namespace cpt {

class CudaGeometry
{
public:
    virtual ~CudaGeometry() = default;
    virtual void unpack_geometry(std::vector<CudaGeometry*>& storage);
    virtual bool unpacked_has_self() const { return true; }
    virtual CudaAABB create_bounding_box() const = 0;
};

inline void CudaGeometry::unpack_geometry(std::vector<CudaGeometry*>& storage)
{
    storage.push_back(this);
}

}
