#include "vertex_container.h"

namespace cpt {

VertexContainer::VertexContainer(
    Eigen::Array3Xf inPositions,
    Eigen::Array2Xf inUvs,
    Eigen::Array3Xf inNormals):
    positions(inPositions),
    uvs(inUvs),
    normals(inNormals)
{
}

}
