#include "cuda_aabb.h"

namespace cpt {

void CudaAABB::expand(const CudaVector<float,3>& other) {
    _min_corner = min(_min_corner, other);
    _max_corner = max(_max_corner, other);
    update_centroid();
}

void CudaAABB::expand(const CudaAABB& other) {
    expand(other.min_corner());
    expand(other.max_corner());
}

void CudaAABB::update_centroid() {
    _centroid = (_min_corner + _max_corner) / 2.0;
}

}
