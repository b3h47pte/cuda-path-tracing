#include "triangle.h"

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

}
