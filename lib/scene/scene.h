#pragma once

#include <memory>
#include "scene/camera/camera.h"
#include "scene/geometry/geometry.h"
#include <unordered_map>
#include <vector>

namespace cpt {

class Scene {
public:
    Scene(
        std::vector<GeometryPtr>&& geometry,
        std::unordered_map<std::string, CameraPtr>&& cameras);

    size_t num_geometry() const { return _geometry.size(); }
    const GeometryPtr& geometry(size_t idx) const { return _geometry[idx]; }

    std::vector<CudaGeometry*> cuda_geometry() const;

    size_t num_cameras() const { return _cameras.size(); }
    bool has_camera(const std::string& id) const { return _cameras.find(id) != _cameras.end(); }
    const CameraPtr& camera(const std::string& id) const { return _cameras.at(id); }

private:
    std::vector<GeometryPtr> _geometry;
    std::unordered_map<std::string, CameraPtr> _cameras;
};

using ScenePtr = std::shared_ptr<Scene>;

}
