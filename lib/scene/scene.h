#pragma once

#include <memory>
#include "scene/geometry/geometry.h"
#include <vector>

namespace cpt {

class Scene {
public:
    Scene(std::vector<GeometryPtr>&& geometry);

    size_t num_geometry() const { return _geometry.size(); }
    const GeometryPtr& geometry(size_t idx) const { return _geometry[idx]; }

private:
    std::vector<GeometryPtr> _geometry;
};

using ScenePtr = std::shared_ptr<Scene>;

}
