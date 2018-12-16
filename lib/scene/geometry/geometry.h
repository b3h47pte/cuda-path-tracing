#pragma once

#include "scene/object.h"
#include "scene/geometry/vertex_container.h"

namespace cpt {

class Geometry: public Object
{
};

using GeometryPtr = std::shared_ptr<Geometry>;

}
