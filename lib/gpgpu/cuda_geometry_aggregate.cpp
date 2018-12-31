#include "cuda_geometry_aggregate.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

CudaGeometryAggregate::CudaGeometryAggregate(
    const std::vector<CudaGeometry*>& children,
    MemoryOwnership ownership):
    _ownership(ownership),
    _num_children(children.size()) {
    CHECK_CUDA_ERROR(cudaMallocManaged(&_children, sizeof(CudaGeometry*) * children.size()));
    for (size_t i = 0; i < _num_children; ++i) {
        _children[i] = children[i];
    }

    set_aabb(create_aabb());
}

CudaGeometryAggregate::~CudaGeometryAggregate() {
    if (_ownership == MemoryOwnership::OWN) {
        for (size_t i = 0; i < _num_children; ++i) {
            cuda_delete(_children[i]);
        }
    }
    cudaFree(_children);
}

void CudaGeometryAggregate::unpack_geometry(std::vector<CudaGeometry*>& storage) {
    // Can't unpack if we don't memory since we can't transfer ownership to 
    // the output storage.
    CHECK_AND_THROW_ERROR(_ownership == MemoryOwnership::OWN, "CudaGeometryAggregate needs to own memory for unpacking.");

    // Transfer ownership of children to the output storage.
    for (size_t i = 0; i < _num_children; ++i) {
        _children[i]->unpack_geometry(storage); 
        if (!_children[i]->unpacked_has_self()) {
            cuda_delete(_children[i]);
        }
    }
    _num_children = 0;
}

CudaAABB CudaGeometryAggregate::create_aabb() const {
    CudaAABB aabb;
    for (size_t i = 0; i < num_children(); ++i) {
        aabb.expand(_children[i]->bounding_box());
    }
    return aabb;
}

}
