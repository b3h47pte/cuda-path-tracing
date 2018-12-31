#pragma once

#include "gpgpu/cuda_geometry.h"
#include "gpgpu/math/cuda_vector.h"
#include <vector>

namespace cpt {

class CudaTriangle: public CudaGeometry
{
public:
    CudaTriangle(
        const std::vector<CudaVector<float,3>*>& vertices,
        const std::vector<CudaVector<float,3>*>& normals,
        const std::vector<CudaVector<float,2>*>& uvs);

private:
    CudaAABB create_aabb() const;

    CudaVector<float,3>* _vertices[3]; 
    CudaVector<float,3>* _normals[3]; 
    CudaVector<float,2>* _uvs[3]; 
};

}
