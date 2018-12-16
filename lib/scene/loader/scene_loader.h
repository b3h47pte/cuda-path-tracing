#pragma once

#include "scene/geometry/geometry.h"
#include "scene/scene.h"
#include <string>
#include <vector>

namespace cpt {

ScenePtr load_scene_from_json(const std::string& fname);

// Stateful SceneBuilder.
class SceneBuilder
{
public:
    ScenePtr construct();

    void add_geometry(const GeometryPtr& geometry);

private:
    std::vector<GeometryPtr> _geometry;
};

}
