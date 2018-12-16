#pragma once

#include <Eigen/Core>
#include <memory>

namespace cpt {

struct VertexContainer
{
    Eigen::ArrayX3f positions;
    Eigen::ArrayX2f uvs;
    Eigen::ArrayX3f normals;
};

using VertexContainerPtr = std::shared_ptr<VertexContainer>;

}
