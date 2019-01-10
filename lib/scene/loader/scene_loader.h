#pragma once

#include <boost/filesystem.hpp>
#include <json/json.hpp>
#include "math/transform.h"
#include <memory>
#include "scene/camera/camera.h"
#include "scene/geometry/geometry.h"
#include "scene/loader/camera_loader.h"
#include "scene/loader/light_loader.h"
#include "scene/loader/mesh_loader.h"
#include "scene/loader/xform_loader.h"
#include "scene/scene.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace cpt {

// Stateful SceneBuilder.
class SceneBuilder
{
public:
    virtual ~SceneBuilder() = default;

    virtual ScenePtr construct();

    virtual void add_geometry(const GeometryPtr& geometry);
    virtual void add_camera(const std::string& id, const CameraPtr& camera);
    virtual void add_light(const LightPtr& light);

    // Transform stack.
    virtual void push_transform(const Transform& xform);
    virtual void pop_transform();
    const Transform& current_transform() const { return _current_xform; }

private:
    std::vector<GeometryPtr> _geometry;
    std::unordered_map<std::string, CameraPtr> _cameras;

    std::vector<Transform> _xform_stack;
    Transform _current_xform;

    std::vector<LightPtr> _lights;

    void update_current_xform();
};

using SceneBuilderPtr = std::shared_ptr<SceneBuilder>;

class SceneLoader
{
public:
    struct SceneLoaderDependencies
    {
        SceneBuilderPtr scene_builder;
        MeshLoaderPtr mesh_loader;
        CameraLoaderPtr camera_loader;
        XformLoaderPtr xform_loader;
        LightLoaderPtr light_loader;

        SceneLoaderDependencies();
    };

    SceneLoader(const SceneLoaderDependencies& deps=SceneLoaderDependencies());
    virtual ~SceneLoader() = default;

    ScenePtr load_scene_from_json(const nlohmann::json& jobj, const std::string& base_dir);
private:
    SceneLoaderDependencies _deps;

    virtual void load_json_mesh(const boost::filesystem::path& base_path, const nlohmann::json& jobj);
    virtual void load_json_camera(const nlohmann::json& jobj);
    virtual void load_json_xform(const nlohmann::json& jobj);
    virtual void load_json_light(const nlohmann::json& jobj);
    virtual void load_json_scene_object_hierarchy(const boost::filesystem::path& base_path, const nlohmann::json& jobj);
    
};

}
