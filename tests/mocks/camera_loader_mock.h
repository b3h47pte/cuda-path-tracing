#pragma once

#include <gmock/gmock.h>
#include <scene/loader/camera_loader.h>

class MockCameraLoader: public cpt::CameraLoader
{
public:
    MOCK_METHOD1(load_camera_from_json, cpt::CameraPtr(const nlohmann::json&));
};
