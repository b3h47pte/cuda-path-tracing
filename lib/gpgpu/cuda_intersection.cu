#include "cuda_intersection.h"

#include "gpgpu/cuda_aabb.h"
#include "gpgpu/cuda_bvh.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_triangle.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

CUDA_DEVHOST bool ray_geometry_intersect(const CudaRay* ray, const CudaGeometry* geometry, CudaIntersection* out_intersection) {
    switch (geometry->type()) {
    case CudaGeometry::Type::BVH:
        return ray_bvh_intersect(ray, static_cast<const CudaBVH*>(geometry), out_intersection);
    case CudaGeometry::Type::Triangle:
        return ray_triangle_intersect(ray, static_cast<const CudaTriangle*>(geometry), out_intersection);
    default:
        break;
    }
    return false;
}

CUDA_DEVHOST bool ray_triangle_intersect(const CudaRay* ray, const CudaTriangle* triangle, CudaIntersection* out_intersection) {
    // Fast, Minimum Storage Ray/Triangle Intersection (Moller and Trumbore 2005).
    // O: ray origin
    // D: ray direction
    // t: amount travelled along ray until intersection
    // TODO: Transform to object space.
    const CudaVector<float,3> O = ray->origin();
    const CudaVector<float,3> D = ray->direction();

    // The triangle equation given barycentric coordinates (u,v) is:
    // T(u,v) = (1 - u - v)V0 + uV1 + vV2
    const CudaVector<float,3>& V0 = triangle->vertex(0);
    const CudaVector<float,3>& V1 = triangle->vertex(1);
    const CudaVector<float,3>& V2 = triangle->vertex(2);

    // Compute the two edges of the triangle: E1 = V1 - V0 and E2 = V2 - V0
    // T = O - V0 because authors suck at not reusing math variable names.
    const CudaVector<float,3> E1 = V1 - V0;
    const CudaVector<float,3> E2 = V2 - V0;
    const CudaVector<float,3> T = O - V0;

    // Equation 6.
    // Note that P = D x E2 and Q = T x E1.
    const CudaVector<float,3> P = cross(D, E2);
    const CudaVector<float,3> Q = cross(T, E1);

    const float PoE1 = dot(P, E1);
    if (abs(PoE1) < CUDA_EPSILON) {
        return false;
    }

    const float u = dot(P, T) / PoE1;
    if (u < 0.f || u > 1.f) {
        return false;
    }

    const float v = dot(Q, D) / PoE1;
    if (v < 0.f || v > 1.f) {
        return false;
    }

    const float t = dot(Q, E2) / PoE1;
    if (t < 0.f || t > ray->max_t()) {
        return false;
    }

    if (out_intersection) {
        out_intersection->hit_geometry = triangle;
        out_intersection->hit_t = t;
    }

    return true;
}

CUDA_DEVHOST bool ray_bvh_intersect(const CudaRay* ray, const CudaBVH* bvh, CudaIntersection* out_intersection) {
    return false;
}

CUDA_DEVHOST bool ray_aabb_intersect(const CudaRay* ray, const CudaAABB* aabb, CudaIntersection* out_intersection) {
    return false;
}

}
