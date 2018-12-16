#include "scene/geometry/geometry_aggregate.h"

#include "scene/geometry/triangle.h"
#include "utilities/eigen_utility.h"

namespace cpt {

GeometryAggregate::GeometryAggregate(
    const VertexContainerPtr& vertex_container,
    const std::vector<GeometryPtr>& children):
    _vertex_container(vertex_container),
    _children(children)
{

}

GeometryAggregatePtr GeometryAggregateBuilder::construct() const
{
    auto vertex_container = std::make_shared<VertexContainer>();
    vertex_container->positions = stl_vector_to_eigen_array(_pos);
    vertex_container->uvs = stl_vector_to_eigen_array(_uv);
    vertex_container->normals = stl_vector_to_eigen_array(_normals);

    std::vector<GeometryPtr> children;
    children.reserve(_face_pos_idx.size());
    for (size_t i = 0; i < _face_pos_idx.size(); ++i) {
        children.emplace_back(std::make_shared<Triangle>(
            vertex_container,
            _face_pos_idx[i],
            _face_uv_idx[i],
            _face_normal_idx[i]));
    }

    auto geometry = std::make_shared<GeometryAggregate>(
        vertex_container,
        children);
    return geometry;
}

void GeometryAggregateBuilder::add_vertex_position(const Eigen::Vector3f& pos)
{
    _pos.push_back(pos);
}

void GeometryAggregateBuilder::add_vertex_uv(const Eigen::Vector2f& uv)
{
    _uv.push_back(uv);
}

void GeometryAggregateBuilder::add_vertex_normal(const Eigen::Vector3f& normal)
{
    _normals.push_back(normal);
}

void GeometryAggregateBuilder::add_face(
    const Eigen::Vector3i& vertex_indices,
    const Eigen::Vector3i& uv_indices,
    const Eigen::Vector3i& normal_indices)
{
    _face_pos_idx.push_back(vertex_indices);
    _face_uv_idx.push_back(uv_indices);
    _face_normal_idx.push_back(normal_indices);
}

}
