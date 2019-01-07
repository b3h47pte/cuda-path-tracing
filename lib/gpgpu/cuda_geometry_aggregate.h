#pragma once

#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_utils.h"
#include "utilities/noncopyable.h"
#include "utilities/memory_ownership.h"
#include <vector>

namespace cpt {

class CudaGeometryAggregate: public CudaGeometry, public Noncopyable
{
public:
    CudaGeometryAggregate(
        const std::vector<CudaGeometry*>& children,
        MemoryOwnership ownership = MemoryOwnership::OWN);
    ~CudaGeometryAggregate();

    void unpack_geometry(std::vector<CudaGeometry*>& storage) override;
    bool unpacked_has_self() const override { return false; }

    CUDA_DEVHOST size_t num_children() const { return _num_children; }
    CUDA_DEVHOST CudaGeometry* child(size_t i) const { return _children[i]; }

    void bake_from_object(const Object& object) override;

private:
    CudaAABB create_aabb() const;

    CudaGeometry** _children;
    MemoryOwnership _ownership;
    size_t         _num_children;
};

}
