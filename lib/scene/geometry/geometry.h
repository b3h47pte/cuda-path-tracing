#pragma once

#include "scene/object.h"
#include "scene/geometry/vertex_container.h"

namespace cpt {

class Geometry: public Object
{

protected:
    VertexContainerPtr _vertex_container;;
};

using GeometryPtr = std::shared_ptr<Geometry>;

}
