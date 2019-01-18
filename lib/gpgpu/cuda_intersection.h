#pragma once

#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaAccelerationStructure;
class CudaGeometry;
class CudaAABB;
class CudaTriangle;
class CudaBVH;
class CudaRay;

enum class IntersectionError
{
    None,
    StackOutOfBounds
};

struct CudaIntersection
{
    const CudaGeometry* hit_geometry{nullptr};
    CudaVector<float,2> hit_uv;
    float hit_t{0.f};
    IntersectionError error{IntersectionError::None};

    CUDA_DEVHOST void register_hit(const CudaGeometry* geom, float t, const CudaVector<float,2>& uv);
    CUDA_DEVHOST CudaVector<float,3> hit_normal() const;
};

CUDA_DEVHOST bool ray_geometry_intersect(const CudaRay* ray, const CudaGeometry* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_accel_structure_intersect(const CudaRay* ray, const CudaAccelerationStructure* accel, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_triangle_intersect(const CudaRay* ray, const CudaTriangle* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_bvh_intersect(const CudaRay* ray, const CudaBVH* geometry, CudaIntersection* out_intersection);

CUDA_DEVHOST bool ray_aabb_intersect(const CudaRay* ray, const CudaAABB* geometry);

}
