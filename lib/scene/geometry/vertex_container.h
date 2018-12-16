#pragma once

#include <Eigen/Core>
#include <memory>

namespace cpt {

struct VertexContainer
{
    Eigen::Array3Xf positions;
    Eigen::Array2Xf uvs;
    Eigen::Array3Xf normals;

    int num_positions() const { return positions.cols(); }
    int num_uvs() const { return uvs.cols(); }
    int num_normals() const { return normals.cols(); }

    Eigen::Ref<const Eigen::Vector3f> position(size_t idx) const { return positions.col(idx); }
    Eigen::Ref<const Eigen::Vector2f> uv(size_t idx) const { return uvs.col(idx); }
    Eigen::Ref<const Eigen::Vector3f> normal(size_t idx) const { return normals.col(idx); }
};

using VertexContainerPtr = std::shared_ptr<VertexContainer>;

}
