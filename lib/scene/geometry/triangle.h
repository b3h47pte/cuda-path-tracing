#pragma once

#include "scene/geometry/vertex_geometry.h"

namespace cpt {

class Triangle: public VertexGeometry
{
public:
    Triangle(
        const VertexContainerPtr& vertex_container,
        const Eigen::Vector3i& vertex_indices,
        const Eigen::Vector3i& uv_indices,
        const Eigen::Vector3i& normal_indices);

    const Eigen::Vector3i& get_vertex_indices() const { return _vertex_indices; }
    const Eigen::Vector3i& get_uv_indices() const { return _uv_indices; }
    const Eigen::Vector3i& get_normal_indices() const { return _normal_indices; }

private:
    Eigen::Vector3i _vertex_indices;
    Eigen::Vector3i _uv_indices;
    Eigen::Vector3i _normal_indices;
};

}