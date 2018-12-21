#include "camera_loader.h"
#include "scene/camera/pinhole_perspective_camera.h"
#include "utilities/error.h"

namespace cpt {
namespace {

const std::string PINHOLE_PERSPECTIVE = "pinhole_perspective";

CameraPtr load_pinhole_perspective_camera_from_json(const nlohmann::json& jobj) {
    auto camera = std::make_shared<PinholePerspectiveCamera>();
    
    auto fov_it = jobj.find("horizontal_fov");
    CHECK_AND_THROW_ERROR(fov_it != jobj.end(), "No FOV specified for pinhole perspective camera.");

    auto ar_it = jobj.find("film_aspect_ratio");
    CHECK_AND_THROW_ERROR(ar_it != jobj.end(), "No aspect ratio specified for pinhole perspective camera.");

    auto focal_length_it = jobj.find("focal_length_mm");
    CHECK_AND_THROW_ERROR(focal_length_it != jobj.end(), "No focal length specified for pinhole perspective camera.");

    auto near_z_it = jobj.find("near_z");
    CHECK_AND_THROW_ERROR(near_z_it != jobj.end(), "No near Z plane specified for pinhole perspective camera.");

    auto far_z_it = jobj.find("far_z");
    CHECK_AND_THROW_ERROR(far_z_it != jobj.end(), "No far Z plane specified for pinhole perspective camera.");

    return camera;
}

}

CameraPtr load_camera_from_json(const nlohmann::json& jobj) {
    auto type_it = jobj.find("type");
    CHECK_AND_THROW_ERROR(type_it != jobj.end(), "No type specified for camera.");

    if (*type_it == PINHOLE_PERSPECTIVE) {
        return load_pinhole_perspective_camera_from_json(jobj);
    } else {
        THROW_ERROR("Invalid camera type: " << *type_it);
    }
    
    return nullptr;
}

}
