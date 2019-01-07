#include "cuda_aabb.h"

namespace cpt {

void CudaAABB::expand(const CudaVector<float,3>& other) {
    if (_empty) {
        _min_corner = other;
        _max_corner = other;
    } else {
        _min_corner = min(_min_corner, other);
        _max_corner = max(_max_corner, other);
    }
    _empty = false;
    update_centroid();
}

void CudaAABB::expand(const CudaAABB& other) {
    expand(other.min_corner());
    expand(other.max_corner());
}

void CudaAABB::update_centroid() {
    _centroid = (_min_corner + _max_corner) / 2.0;
}

CudaAABB CudaAABB::transform(const CudaAffineTransform& xform) const {
    CudaAABB ret_box;

    CudaVector<float,3> object_corners[8];
    corners(object_corners);

    for (auto i = 0; i < 8; ++i) {
        ret_box.expand(xform.transform(object_corners[i], true));
    }

    return ret_box;
}

void CudaAABB::corners(CudaVector<float,3> corners[8]) const {
    int counter = 0;
    for (int i = 0 ; i < 2; ++i) {
        for (int j = 0 ; j < 2; ++j) {
            for (int k = 0 ; k < 2; ++k) {
                corners[counter][0] = (i == 0) ? _min_corner[0] : _max_corner[0];
                corners[counter][1] = (i == 0) ? _min_corner[1] : _max_corner[1];
                corners[counter][2] = (i == 0) ? _min_corner[2] : _max_corner[2];
                ++counter;
            }
        }
    }
}

}
