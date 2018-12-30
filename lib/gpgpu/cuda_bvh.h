#pragma once 

#include "gpgpu/cuda_acceleration_structure.h"

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

private:
    // Current node information.
    // If we're a leaf node, actual geometry is stored in
    // CudaAccelerationStructure::_aggregate.
    CudaBVH** _bvh_children;
    size_t    _max_num_children;
};

}
