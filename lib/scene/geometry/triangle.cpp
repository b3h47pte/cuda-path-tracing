#include "triangle.h"

#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_triangle.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

Triangle::Triangle(
    const VertexContainerPtr& vertex_container,
    const Eigen::Vector3i& vertex_indices,
    const Eigen::Vector3i& uv_indices,
    const Eigen::Vector3i& normal_indices):
    VertexGeometry(vertex_container),
    _vertex_indices(vertex_indices),
    _uv_indices(uv_indices),
    _normal_indices(normal_indices)
{

}

CudaGeometry* Triangle::create_cuda(CudaGeometryCache& cache) const {
    auto* key = const_cast<Triangle*>(this);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    std::vector<CudaVector<float,3>> positions(3);
    std::vector<CudaVector<float,3>> normals(3);
    std::vector<CudaVector<float,2>> uvs(3);

    for (auto i = 0; i < 3; ++i) {
        positions[i] = eigen_vector_to_cuda<float,3>(get_vertex_container()->position(_vertex_indices(i)));
        normals[i] = eigen_vector_to_cuda<float,3>(get_vertex_container()->normal(_normal_indices(i)));
        uvs[i] = eigen_vector_to_cuda<float,2>(get_vertex_container()->uv(_uv_indices(i)));
    }

    auto* ptr = cuda_new<CudaTriangle>(positions, normals, uvs);
    cache[key] = ptr;
    return ptr;
}

}
