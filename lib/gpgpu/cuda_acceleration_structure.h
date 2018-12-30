#pragma once

#include "gpgpu/cuda_geometry.h"
#include <vector>

namespace cpt {

class CudaAccelerationStructure: public CudaGeometry
{
public:
    CudaAccelerationStructure(const std::vector<CudaGeometry*>& cuda_geom);
    ~CudaAccelerationStructure();

private:
    CudaGeometry** _cuda_geom;
};

}
