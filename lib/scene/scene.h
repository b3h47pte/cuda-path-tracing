#pragma once

#include <memory>
#include "scene/camera/camera.h"
#include "scene/geometry/geometry.h"
#include "scene/lights/light.h"
#include <unordered_map>
#include <vector>

namespace cpt {

class Scene {
public:
    Scene(
        std::vector<GeometryPtr>&& geometry,
        std::unordered_map<std::string, CameraPtr>&& cameras,
        std::vector<LightPtr>&& lights);

    size_t num_geometry() const { return _geometry.size(); }
    const GeometryPtr& geometry(size_t idx) const { return _geometry[idx]; }

    size_t num_cameras() const { return _cameras.size(); }
    bool has_camera(const std::string& id) const { return _cameras.find(id) != _cameras.end(); }
    const CameraPtr& camera(const std::string& id) const { return _cameras.at(id); }

    size_t num_lights() const { return _lights.size(); }
    const LightPtr& light(size_t idx) const { return _lights[idx]; }

private:
    std::vector<GeometryPtr> _geometry;
    std::unordered_map<std::string, CameraPtr> _cameras;
    std::vector<LightPtr> _lights;
};

using ScenePtr = std::shared_ptr<Scene>;

}
