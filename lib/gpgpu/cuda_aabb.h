#pragma once

#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaAABB
{
public:
    void expand(const CudaVector<float,3>& other);
    void expand(const CudaAABB& other);

    const CudaVector<float,3>& centroid() const { return _centroid; }
    const CudaVector<float,3>& min_corner() const { return _min_corner; }
    const CudaVector<float,3>& max_corner() const { return _max_corner; }

private:
    void update_centroid();

    CudaVector<float,3> _centroid;
    CudaVector<float,3> _min_corner;
    CudaVector<float,3> _max_corner;
};

}
