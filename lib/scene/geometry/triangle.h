#pragma once

#include "scene/geometry/geometry.h"
#include "scene/geometry/vertex_container.h"

namespace cpt {

class Triangle: public Geometry
{
public:
    Triangle(
        const VertexContainerPtr& vertex_container,
        const Eigen::Vector3i& vertex_indices,
        const Eigen::Vector3i& uv_indices,
        const Eigen::Vector3i& normal_indices);

private:
    VertexContainerPtr _vertex_container;
    Eigen::Vector3i _vertex_indices;
    Eigen::Vector3i _uv_indices;
    Eigen::Vector3i _normal_indices;
};

}
