#include "scene_loader.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <json/json.hpp>
#include "scene/loader/mesh_loader.h"
#include "utilities/error.h"

namespace bfs = boost::filesystem;

namespace cpt {
namespace {

void load_json_mesh(const bfs::path& base_path, const nlohmann::json& jobj, SceneBuilder& builder) {
    auto filenameIt = jobj.find("filename");
    CHECK_AND_THROW_ERROR(filenameIt != jobj.end(), "Invalid mesh object. No filename specified.");

    GeometryPtr geom = load_mesh_from_file((base_path / bfs::path(filenameIt->get<std::string>())).native());
    builder.add_geometry(geom);
}

void load_json_scene_object_hierarchy(const bfs::path& base_path, const nlohmann::json& jobj, SceneBuilder& builder) {
    // No type field? Error out.
    auto typeIt = jobj.find("type");
    CHECK_AND_THROW_ERROR(typeIt != jobj.end(), "Invalid scene object. No type specified.");

    // Check type and dispatch to appropriate function.
    if (*typeIt == "mesh") {
        load_json_mesh(base_path, jobj, builder);
    } else if (*typeIt == "root") {
        // Can safely ignore the root.
    } else {
        THROW_ERROR("Hierarchy object type [" << *typeIt << "] is not supported.");
    }

    // Handle children.
    auto childrenIt = jobj.find("children");
    if (childrenIt != jobj.end()) {
        for (const auto& child : *childrenIt) {
            load_json_scene_object_hierarchy(base_path, child, builder);
        }
    }
}

}

ScenePtr load_scene_from_json(const std::string& fname)
{
    CHECK_AND_THROW_ERROR(bfs::exists(fname), "Scene JSON does not exists [" << fname << "].");
    nlohmann::json jobj;

    const bfs::path base_path = bfs::path(fname).parent_path();
    std::ifstream fs(fname);
    fs >> jobj;

    SceneBuilder builder;

    // Load in scene object hierarchy (geometry, cameras, lights).
    load_json_scene_object_hierarchy(base_path, jobj, builder);

    fs.close();
    return builder.construct();
}

ScenePtr SceneBuilder::construct() {
    auto scene = std::make_shared<Scene>(
        std::move(_geometry));
    return scene;
}

void SceneBuilder::add_geometry(const GeometryPtr& geometry) {
    _geometry.push_back(geometry);
}

}
