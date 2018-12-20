#pragma once

#include "math/transform.h"
#include "scene/camera/camera.h"
#include "scene/geometry/geometry.h"
#include "scene/scene.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace cpt {

ScenePtr load_scene_from_json(const std::string& fname);

// Stateful SceneBuilder.
class SceneBuilder
{
public:
    ScenePtr construct();

    void add_geometry(const GeometryPtr& geometry);
    void add_camera(const std::string& id, const CameraPtr& camera);

    // Transform stack.
    void push_transform(const Transform& xform);
    void pop_transform();
    const Transform& current_transform() const { return _current_xform; }

private:
    std::vector<GeometryPtr> _geometry;
    std::unordered_map<std::string, CameraPtr> _cameras;

    std::vector<Transform> _xform_stack;
    Transform _current_xform;

    void update_current_xform();
};

}
