#pragma once

#include "gpgpu/cuda_aabb.h"
#include "gpgpu/cuda_object.h"
#include "gpgpu/math/cuda_vector.h"
#include <vector>

namespace cpt {

class CudaGeometry: public CudaObject
{
public:
    virtual ~CudaGeometry() = default;
    virtual void unpack_geometry(std::vector<CudaGeometry*>& storage);
    virtual bool unpacked_has_self() const { return true; }

    const CudaAABB& bounding_box() const { return _aabb; }
    const CudaVector<float,3>& centroid() const { return _aabb.centroid(); }

protected:
    void set_aabb(const CudaAABB& aabb) { _aabb = aabb; }

private:
    CudaAABB _aabb;
};

inline void CudaGeometry::unpack_geometry(std::vector<CudaGeometry*>& storage)
{
    storage.push_back(this);
}

}
