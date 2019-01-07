#include "cuda_acceleration_structure.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"

namespace cpt {

CudaAccelerationStructure::CudaAccelerationStructure(
    Type type,
    const std::vector<CudaGeometry*>& cuda_geom,
    MemoryOwnership ownership):
    CudaGeometry(type),
    _aggregate(cuda_geom, ownership) {

    // This is kinda shitty...
    _aggregate.bake_from_object(Object());
    set_aabb(_aggregate.world_space_bounding_box());
    set_world_space_aabb(_aggregate.world_space_bounding_box());
}

CudaAccelerationStructure::~CudaAccelerationStructure() {
}

}
