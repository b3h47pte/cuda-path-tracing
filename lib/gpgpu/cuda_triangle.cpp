#include "cuda_triangle.h"
#include "utilities/error.h"
#include "utilities/timer.h"

namespace cpt {

CudaTriangle::CudaTriangle(
    const std::vector<CudaVector<float,3>>& vertices,
    const std::vector<CudaVector<float,3>>& normals,
    const std::vector<CudaVector<float,2>>& uvs):
    CudaGeometry(Type::Triangle) {

    CHECK_AND_THROW_ERROR(vertices.size() == 3, "CudaTriangle needs three vertices.");
    CHECK_AND_THROW_ERROR(normals.size() == 3, "CudaTriangle needs three normals.");
    CHECK_AND_THROW_ERROR(uvs.size() == 3, "CudaTriangle needs three UVs.");

    for (auto i = 0; i < 3; ++i) {
        _vertices[i] = vertices[i];
        _normals[i] = normals[i];
        _uvs[i] = uvs[i];
    }

    set_aabb(create_aabb());
}

CudaAABB CudaTriangle::create_aabb() const {
    CudaAABB aabb;
    for (auto i = 0; i < 3; ++i) {
        aabb.expand(_vertices[i]);
    }
    return aabb;
}

}
