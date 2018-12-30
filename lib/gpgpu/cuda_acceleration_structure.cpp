#include "cuda_acceleration_structure.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"

namespace cpt {

CudaAccelerationStructure::CudaAccelerationStructure(
    const std::vector<CudaGeometry*>& cuda_geom,
    MemoryOwnership ownership):
    _aggregate(cuda_geom, ownership) {
}

CudaAccelerationStructure::~CudaAccelerationStructure() {
}

}
