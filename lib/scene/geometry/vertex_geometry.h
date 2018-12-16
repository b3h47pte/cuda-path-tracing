#pragma once

#include "scene/geometry/geometry.h"
#include "scene/geometry/vertex_container.h"

namespace cpt {

class VertexGeometry: public Geometry
{
public:
    VertexGeometry(
        const VertexContainerPtr& vertex_container):
        _vertex_container(vertex_container)
    {}

    const VertexContainerPtr& get_vertex_container() const { return _vertex_container; }

protected:
    VertexContainerPtr _vertex_container;
};

using VertexGeometryPtr = std::shared_ptr<VertexGeometry>;

}
