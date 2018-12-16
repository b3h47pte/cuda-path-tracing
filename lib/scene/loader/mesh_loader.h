#pragma once

#include "scene/geometry/geometry.h"
#include <string>

namespace cpt {

GeometryPtr load_mesh_from_file(const std::string& fname);
GeometryPtr load_obj_from_file(const std::string& fname);

}
