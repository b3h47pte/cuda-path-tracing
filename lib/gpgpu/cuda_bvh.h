#pragma once 

#include "gpgpu/cuda_acceleration_structure.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaBVH: public CudaAccelerationStructure
{
public:
    CudaBVH(
        const std::vector<CudaGeometry*>& cuda_geom,
        size_t max_num_children = 4);

    CudaBVH(
        const std::vector<CudaGeometry*>& cuda_geom,
        size_t max_num_children,
        MemoryOwnership ownership,
        size_t split_axis);

    ~CudaBVH();

    CUDA_DEVHOST bool has_bvh_children() const { return _bvh_children != nullptr; }
    CUDA_DEVHOST size_t max_num_children() const { return _max_num_children; }
    CUDA_DEVHOST CudaBVH* bvh_child(size_t idx) const { return _bvh_children[idx]; }

private:
    // Current node information.
    // If we're a leaf node, actual geometry is stored in
    // CudaAccelerationStructure::_aggregate.
    CudaBVH** _bvh_children;
    size_t    _max_num_children;
};

}
