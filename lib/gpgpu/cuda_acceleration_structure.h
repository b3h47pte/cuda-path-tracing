#pragma once

#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "gpgpu/cuda_utils.h"
#include "utilities/memory_ownership.h"
#include <vector>

namespace cpt {

class CudaAccelerationStructure: public CudaGeometry
{
public:
    CudaAccelerationStructure(
        Type type,
        const std::vector<CudaGeometry*>& cuda_geom,
        MemoryOwnership ownership = MemoryOwnership::OWN);
    ~CudaAccelerationStructure();

    CUDA_DEVHOST const CudaGeometryAggregate& aggregate() const { return _aggregate; }

protected:
    CudaGeometryAggregate _aggregate;
};

}
