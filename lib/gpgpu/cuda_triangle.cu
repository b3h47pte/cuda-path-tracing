#include "cuda_triangle.h"

namespace cpt {

CUDA_DEVHOST CudaVector<float,3> triangle_normal(const CudaTriangle& triangle, float u, float v) {
    // TODO: Smooth normal?
    CudaVector<float,3> N = cross(
        triangle.vertex(1) - triangle.vertex(0),
        triangle.vertex(2) - triangle.vertex(0));
    N.normalize();
    return N;
}

}
