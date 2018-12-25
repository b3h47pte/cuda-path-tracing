#pragma once

#include "scene/geometry/geometry.h"
#include <memory>
#include <string>

namespace cpt {

class MeshLoader
{
public:
    virtual ~MeshLoader() = default;

    virtual GeometryPtr load_mesh_from_file(const std::string& fname);
    virtual GeometryPtr load_obj_from_file(const std::string& fname);
};

using MeshLoaderPtr = std::shared_ptr<MeshLoader>;

}
