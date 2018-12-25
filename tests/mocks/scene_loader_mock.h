#pragma once

#include <gmock/gmock.h>
#include <scene/loader/scene_loader.h>

class MockSceneLoader: public cpt::SceneLoader
{
public:
    MockSceneLoader(const SceneLoaderDependencies& deps):
        cpt::SceneLoader(deps)
    {}

    MOCK_METHOD2(load_json_mesh, void(const boost::filesystem::path&, const nlohmann::json&));
    MOCK_METHOD1(load_json_camera, void(const nlohmann::json&));
    MOCK_METHOD1(load_json_xform, void(const nlohmann::json&));
    MOCK_METHOD2(load_json_scene_object_hierarchy, void(const boost::filesystem::path&, const nlohmann::json&));
};
