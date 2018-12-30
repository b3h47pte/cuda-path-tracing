#include "cuda_bvh.h"
#include "gpgpu/cuda_ptr.h"

namespace cpt {

CudaBVH::CudaBVH(
    const std::vector<CudaGeometry*>& cuda_geom,
    size_t max_num_children):
    CudaBVH(cuda_geom, max_num_children, MemoryOwnership::OWN, 0) {
}

CudaBVH::CudaBVH(
    const std::vector<CudaGeometry*>& cuda_geom,
    size_t max_num_children,
    MemoryOwnership ownership,
    size_t split_axis):
    CudaAccelerationStructure(cuda_geom, ownership),
    _max_num_children(max_num_children) {

    // Construct this node - get bounding box of all
    // the input geometry.
    _bounding_box = create_bounding_box();

    // Check if this node should be a leaf node.
    if (cuda_geom.size() < _max_num_children) {
        _bvh_children = nullptr;
        return;
    }

    // Separate geometry along the split axis.

}

CudaBVH::~CudaBVH() {
    cuda_delete(_bvh_children);
}

CudaAABB CudaBVH::create_bounding_box() const {
    return _aggregate.create_bounding_box();
}

}
