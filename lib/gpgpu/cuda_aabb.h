#pragma once

#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"
#include "gpgpu/math/cuda_affine_transform.h"

namespace cpt {

class CudaAABB
{
public:
    void expand(const CudaVector<float,3>& other);
    void expand(const CudaAABB& other);

    CudaAABB transform(const CudaAffineTransform& xform) const;
    void corners(CudaVector<float,3> corners[8]) const;

    CUDA_DEVHOST const CudaVector<float,3>& centroid() const { return _centroid; }
    CUDA_DEVHOST const CudaVector<float,3>& min_corner() const { return _min_corner; }
    CUDA_DEVHOST const CudaVector<float,3>& max_corner() const { return _max_corner; }

private:
    void update_centroid();

    CudaVector<float,3> _centroid;
    CudaVector<float,3> _min_corner;
    CudaVector<float,3> _max_corner;
    bool _empty{true};
};

}
