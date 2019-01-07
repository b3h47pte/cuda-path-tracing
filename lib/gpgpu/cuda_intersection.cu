#include "cuda_intersection.h"

#include "gpgpu/cuda_aabb.h"
#include "gpgpu/cuda_bvh.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_stack.h"
#include "gpgpu/cuda_triangle.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_affine_transform.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

CUDA_DEVHOST void CudaIntersection::register_hit(const CudaGeometry* geom, float t) {
    if (hit_geometry && t > hit_t) {
        return;
    }
    hit_geometry = geom;
    hit_t = t;
}

CUDA_DEVHOST bool ray_geometry_intersect(const CudaRay* ray, const CudaGeometry* geometry, CudaIntersection* out_intersection) {
    switch (geometry->type()) {
    case CudaGeometry::Type::Triangle:
        return ray_triangle_intersect(ray, static_cast<const CudaTriangle*>(geometry), out_intersection);
    default:
        break;
    }
    return false;
}

CUDA_DEVHOST bool ray_accel_structure_intersect(const CudaRay* ray, const CudaAccelerationStructure* accel, CudaIntersection* out_intersection) {
    switch (accel->type()) {
    case CudaGeometry::Type::BVH:
        return ray_bvh_intersect(ray, static_cast<const CudaBVH*>(accel), out_intersection);
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
    const CudaAffineTransform& world_to_object_xform = triangle->world_to_object_xform();
    const CudaVector<float,3> O = world_to_object_xform.transform(ray->origin(), true);
    const CudaVector<float,3> D = world_to_object_xform.transform(ray->direction(), false);

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
    if (v < 0.f || u + v > 1.f) {
        return false;
    }

    const float t = dot(Q, E2) / PoE1;
    if (t < 0.f || t > ray->max_t()) {
        return false;
    }

    if (out_intersection) {
        out_intersection->register_hit(triangle, t);
    }

    return true;
}

CUDA_DEVHOST bool ray_bvh_intersect(const CudaRay* ray, const CudaBVH* bvh, CudaIntersection* out_intersection) {
    // TODO: Stackless BVH traversal?
    // Only keep BVHs who passed the bounding box intersection test to minimize
    // the use of the stack as much as possible.
    CudaStack<const CudaBVH*> bvh_stack;

    if (ray_aabb_intersect(ray, &bvh->world_space_bounding_box())) {
        bvh_stack.push(bvh);
    }

    bool success = false;

    while (!bvh_stack.empty()) {
        const CudaBVH* current_bvh = bvh_stack.pop();
        if (current_bvh->has_bvh_children()) {
            for (size_t i = 0; i < current_bvh->max_num_children(); ++i) {
                const CudaBVH* child = current_bvh->bvh_child(i);
                if (ray_aabb_intersect(ray, &child->world_space_bounding_box())) {
                    bvh_stack.push(child);
                }
            }
        } else {
            // Figure out which leaf we hit.
            const CudaGeometryAggregate& aggregate = current_bvh->aggregate();
            for (size_t i = 0; i < aggregate.num_children(); ++i) {
                success |= ray_geometry_intersect(
                    ray,
                    aggregate.child(i),
                    out_intersection);
            }
        }
    }

    if (bvh_stack.out_of_bounds()) {
        if (out_intersection) {
            out_intersection->error = IntersectionError::StackOutOfBounds;
        }
        return false;
    }
    return success;
}

CUDA_DEVHOST bool ray_aabb_intersect(const CudaRay* ray, const CudaAABB* aabb) {
    // Slab test in all 3 dimensions.
    const CudaVector<float,3>& O = ray->origin();
    const CudaVector<float,3>& D = ray->direction();

    const CudaVector<float,3>& min_corner = aabb->min_corner();
    const CudaVector<float,3>& max_corner = aabb->max_corner();

    bool has_range = false;
    float min_range = 0.f;
    float max_range = 0.f;

    for (auto i = 0; i < 3; ++i) {
        // If we aren't moving in some direction, then we can't intersect in that direction
        // unless we're already inside. Leave the job of determining intersection to
        // another axis.
        if (D[i] == 0.f) {
             if (O[i] < min_corner[i] || O[i] > max_corner[i]) {
                return false;
            }
            continue;
        }

        float min_t = (min_corner[i] - O[i]) / D[i];
        float max_t = (max_corner[i] - O[i]) / D[i];

        if (max_t < min_t) {
            const float tmp = min_t;
            min_t = max_t;
            max_t = tmp;
        }

        if (has_range) {
            if (min_t > max_range || max_t < min_range) {
                return false;
            }
            min_range = ::max(min_range, min_t);
            max_range = ::min(max_range, max_t);
        } else {
            has_range = true;
            min_range = min_t;
            max_range = max_t;
        }
    }

    return true;
}

}
