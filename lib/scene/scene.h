#pragma once

#include <memory>
#include "scene/geometry/geometry.h"
#include <vector>

namespace cpt {

class Scene {
public:
    Scene(std::vector<GeometryPtr>&& geometry);

private:
    std::vector<GeometryPtr> _geometry;
};

using ScenePtr = std::shared_ptr<Scene>;

}
