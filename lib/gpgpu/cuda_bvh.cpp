#include "cuda_bvh.h"

#include <algorithm>
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/math/cuda_vector.h"

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
    CudaAccelerationStructure(Type::BVH, cuda_geom, ownership),
    _max_num_children(max_num_children) {

    // Check if this node should be a leaf node.
    if (cuda_geom.size() <= _max_num_children) {
        _bvh_children = nullptr;
        return;
    }

    // Create children BVH.
    CHECK_CUDA_ERROR(cudaMallocManaged(&_bvh_children, sizeof(CudaBVH*) * _max_num_children));

    // Sort geometry along the split axis using the centroid.
    std::vector<CudaGeometry*> sorted_geometry(cuda_geom.begin(), cuda_geom.end());
    std::sort(sorted_geometry.begin(), sorted_geometry.end(), 
    [split_axis](CudaGeometry* a, CudaGeometry* b) {
        return a->world_space_bounding_box().centroid()[split_axis] < b->world_space_bounding_box().centroid()[split_axis];
    });

    const size_t num_children_per_node = cuda_geom.size() / max_num_children;
    const size_t num_children_last_node = cuda_geom.size() - num_children_per_node * (max_num_children - 1);

    size_t ptr = 0;
    for (size_t i = 0; i < max_num_children; ++i) {
        const size_t num_to_use = (i == max_num_children - 1) ? num_children_last_node : num_children_per_node;

        std::vector<CudaGeometry*> subset(
            sorted_geometry.begin() + ptr,
            sorted_geometry.begin() + ptr + num_to_use);

        _bvh_children[i] = cuda_new<CudaBVH>(
            subset,
            _max_num_children,
            MemoryOwnership::DO_NOT_OWN,
            (split_axis + 1) % 3);

        ptr += num_to_use;
    }
    assert(ptr == sorted_geometry.size());
}

CudaBVH::~CudaBVH() {
    cuda_delete(_bvh_children);
}

}
