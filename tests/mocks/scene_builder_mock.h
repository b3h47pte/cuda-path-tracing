#pragma once

#include <gmock/gmock.h>
#include <scene/loader/scene_loader.h>

class MockSceneBuilder: public cpt::SceneBuilder
{
public:
    MOCK_METHOD0(construct, cpt::ScenePtr());

    MOCK_METHOD1(add_geometry, void(const cpt::GeometryPtr& geometry));
    MOCK_METHOD2(add_camera, void(const std::string&, const cpt::CameraPtr&));

    MOCK_METHOD1(push_transform, void(const cpt::Transform&));
    MOCK_METHOD0(pop_transform, void());
};
