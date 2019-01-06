#pragma once

#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaGeometry;
class CudaAABB;
class CudaTriangle;
class CudaBVH;
class CudaRay;

struct CudaIntersection
{
    const CudaGeometry* hit_geometry{nullptr};
    float hit_t{0.f};
};

CUDA_DEVHOST bool ray_geometry_intersect(const CudaRay* ray, const CudaGeometry* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_triangle_intersect(const CudaRay* ray, const CudaTriangle* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_bvh_intersect(const CudaRay* ray, const CudaBVH* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_aabb_intersect(const CudaRay* ray, const CudaAABB* geometry, CudaIntersection* out_intersection);

}
