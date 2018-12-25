#include "scene_loader.h"
#include <boost/filesystem.hpp>
#include <json/json.hpp>
#include "scene/loader/camera_loader.h"
#include "scene/loader/mesh_loader.h"
#include "scene/loader/xform_loader.h"
#include "utilities/error.h"

namespace bfs = boost::filesystem;

namespace cpt {
namespace {

const std::string MESH_ID = "mesh";
const std::string ROOT_ID = "root";
const std::string CAMERA_ID = "camera";
const std::string TRANSFORM_ID = "xform";

}

ScenePtr SceneBuilder::construct() {
    auto scene = std::make_shared<Scene>(
        std::move(_geometry),
        std::move(_cameras));
    return scene;
}

void SceneBuilder::add_geometry(const GeometryPtr& geometry) {
    geometry->set_object_to_world_xform(current_transform());
    _geometry.push_back(geometry);
}

void SceneBuilder::add_camera(const std::string& id, const CameraPtr& camera) {
    CHECK_AND_THROW_ERROR(_cameras.find(id) == _cameras.end(), "Can not reuse the same camera ID for two or more cameras.");
    camera->set_object_to_world_xform(current_transform());
    _cameras[id] = camera;
}

void SceneBuilder::push_transform(const Transform& xform) {
    _xform_stack.push_back(xform);
    update_current_xform();
}

void SceneBuilder::pop_transform() {
    _xform_stack.erase(_xform_stack.end() - 1);
    update_current_xform();
}

void SceneBuilder::update_current_xform() {
    if (_xform_stack.empty()) {
        _current_xform.reset();
        return;
    }

    _current_xform = _xform_stack[0];
    for (size_t i = 1; i < _xform_stack.size(); ++i) {
        _current_xform *= _xform_stack[i];
    }
}

SceneLoader::SceneLoaderDependencies::SceneLoaderDependencies():
    scene_builder(std::make_shared<SceneBuilder>()),
    mesh_loader(std::make_shared<MeshLoader>()),
    camera_loader(std::make_shared<CameraLoader>()),
    xform_loader(std::make_shared<XformLoader>())
{}

SceneLoader::SceneLoader(const SceneLoaderDependencies& deps):
    _deps(deps)
{}

ScenePtr SceneLoader::load_scene_from_json(const nlohmann::json& jobj, const std::string& base_dir) {
    CHECK_AND_THROW_ERROR(bfs::exists(base_dir), "Base directory does not exist [" << base_dir << "].");
    const bfs::path base_path(base_dir);

    // Load in scene object hierarchy (geometry, cameras, lights).
    load_json_scene_object_hierarchy(base_path, jobj);

    return _deps.scene_builder->construct();
}

void SceneLoader::load_json_mesh(const bfs::path& base_path, const nlohmann::json& jobj) {
    auto filename_it = jobj.find("filename");
    CHECK_AND_THROW_ERROR(filename_it != jobj.end(), "Invalid mesh object. No filename specified.");

    GeometryPtr geom = _deps.mesh_loader->load_mesh_from_file((base_path / bfs::path(filename_it->get<std::string>())).native());
    _deps.scene_builder->add_geometry(geom);
}

void SceneLoader::load_json_camera(const nlohmann::json& jobj) {
    // A unique id must be found for the camera.
    auto id_it = jobj.find("id");
    CHECK_AND_THROW_ERROR(id_it != jobj.end(), "Invalid camera object. No ID specified.");

    auto obj_it = jobj.find("object");
    CHECK_AND_THROW_ERROR(obj_it != jobj.end(), "Invalid camera object. No 'object' specified.");

    CameraPtr camera = _deps.camera_loader->load_camera_from_json(*obj_it);
    _deps.scene_builder->add_camera(*id_it, camera);
}

void SceneLoader::load_json_xform(const nlohmann::json& jobj) {
    auto obj_it = jobj.find("object");
    CHECK_AND_THROW_ERROR(obj_it != jobj.end(), "Invalid transform object. No 'object' specified.");

    Transform xform = _deps.xform_loader->load_xform_from_json(*obj_it);
    _deps.scene_builder->push_transform(xform);
}

void SceneLoader::load_json_scene_object_hierarchy(const bfs::path& base_path, const nlohmann::json& jobj) {
    // No type field? Error out.
    auto type_it = jobj.find("type");
    CHECK_AND_THROW_ERROR(type_it != jobj.end(), "Invalid scene object. No type specified.");

    // Check type and dispatch to appropriate function.
    if (*type_it == MESH_ID) {
        load_json_mesh(base_path, jobj);
    } else if (*type_it == ROOT_ID) {
        // Can safely ignore the root.
    } else if (*type_it == CAMERA_ID) {
        load_json_camera(jobj);
    } else if (*type_it == TRANSFORM_ID) {
        load_json_xform(jobj);
    } else {
        THROW_ERROR("Hierarchy object type [" << *type_it << "] is not supported.");
    }

    // Handle children.
    auto children_it = jobj.find("children");
    if (children_it != jobj.end()) {
        for (const auto& child : *children_it) {
            load_json_scene_object_hierarchy(base_path, child);
        }
    }

    // TODO: Maybe this fits better in the SceneBuilder having a more generic
    //       start/stop transaction sort of thing.
    // Post-process only to handle popping the xform.
    if (*type_it == TRANSFORM_ID) {
        _deps.scene_builder->pop_transform();
    }
}

}
