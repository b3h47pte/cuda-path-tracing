#pragma once

#include <Eigen/Core>
#include "scene/geometry/geometry.h"
#include "scene/geometry/vertex_container.h"
#include <vector>

namespace cpt {

class GeometryAggregate: public Geometry
{
public:
    GeometryAggregate(const std::vector<GeometryPtr>& children);

    size_t num_children() const { return _children.size(); }
    const GeometryPtr& get_geometry(size_t idx) const { return _children[idx]; }

private:
    std::vector<GeometryPtr> _children;
};

using GeometryAggregatePtr = std::shared_ptr<GeometryAggregate>;

class GeometryAggregateBuilder
{
public:
    GeometryAggregatePtr construct() const; 

    void add_vertex_position(const Eigen::Vector3f& pos);
    void add_vertex_uv(const Eigen::Vector2f& uv);
    void add_vertex_normal(const Eigen::Vector3f& normal);

    void add_face(const Eigen::Vector3i& vertex_indices,
                  const Eigen::Vector3i& uv_indices,
                  const Eigen::Vector3i& normal_indices);

private:
    std::vector<Eigen::Vector3f> _pos;
    std::vector<Eigen::Vector2f> _uv;
    std::vector<Eigen::Vector3f> _normals;

    std::vector<Eigen::Vector3i> _face_pos_idx;
    std::vector<Eigen::Vector3i> _face_uv_idx;
    std::vector<Eigen::Vector3i> _face_normal_idx;
};

}
