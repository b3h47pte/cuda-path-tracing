#pragma once

#include <json/json.hpp>
#include "scene/camera/camera.h"

namespace cpt {

CameraPtr load_camera_from_json(const nlohmann::json& jobj);

}
