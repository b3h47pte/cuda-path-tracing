#pragma once

#include <json/json.hpp>
#include <memory>
#include "scene/camera/camera.h"

namespace cpt {

class CameraLoader
{
public:
    virtual ~CameraLoader() = default;
    virtual CameraPtr load_camera_from_json(const nlohmann::json& jobj);

private:
    virtual CameraPtr load_pinhole_perspective_camera_from_json(const nlohmann::json& jobj);
};

using CameraLoaderPtr = std::shared_ptr<CameraLoader>;

}
