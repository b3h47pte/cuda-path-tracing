#pragma once

#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "utilities/memory_ownership.h"
#include <vector>

namespace cpt {

class CudaAccelerationStructure: public CudaGeometry
{
public:
    CudaAccelerationStructure(
        const std::vector<CudaGeometry*>& cuda_geom,
        MemoryOwnership ownership = MemoryOwnership::OWN);
    ~CudaAccelerationStructure();

protected:
    CudaGeometryAggregate _aggregate;
};

}