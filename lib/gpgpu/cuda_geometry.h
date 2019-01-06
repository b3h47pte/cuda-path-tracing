#pragma once

#include "gpgpu/cuda_aabb.h"
#include "gpgpu/cuda_object.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"
#include <vector>

namespace cpt {

class CudaGeometry: public CudaObject
{
public:
    enum class Type
    {
        BVH,
        Triangle,
        Aggregate
    };

    CudaGeometry(Type type);
    virtual ~CudaGeometry() = default;
    virtual void unpack_geometry(std::vector<CudaGeometry*>& storage);
    virtual bool unpacked_has_self() const { return true; }

    CUDA_DEVHOST const CudaAABB& bounding_box() const { return _aabb; }
    CUDA_DEVHOST const CudaVector<float,3>& centroid() const { return _aabb.centroid(); }
    CUDA_DEVHOST Type type() const { return _type; }
    CUDA_DEVHOST size_t id() const { return _id; }

protected:
    void set_aabb(const CudaAABB& aabb) { _aabb = aabb; }

private:
    Type     _type;
    CudaAABB _aabb;
    size_t   _id;
};

inline void CudaGeometry::unpack_geometry(std::vector<CudaGeometry*>& storage)
{
    storage.push_back(this);
}

}
