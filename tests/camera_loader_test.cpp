#include <scene/camera/pinhole_perspective_camera.h>
#include <scene/loader/camera_loader.h>
#include <json/json.hpp>
#include "test_common.h"

namespace {
constexpr double epsilon = 1e-6;
}

TEST(CameraLoader,TestLoadNoType)
{
    std::string test_string = R"(
{ 
})";

    cpt::CameraLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    EXPECT_THROW(loader.load_camera_from_json(jobj), std::runtime_error);
}

TEST(CameraLoader,TestLoadUnsupportedType)
{
    std::string test_string = R"(
{ 
    "type": "_THIS_TYPE_NOT_SUPPORTED_"
})";

    cpt::CameraLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    EXPECT_THROW(loader.load_camera_from_json(jobj), std::runtime_error);
}

TEST(CameraLoader,TestLoadIncompletePinholePerspective)
{
    std::string test_string = R"(
{ 
    "type": "pinhole_perspective",
    "horizontal_fov": 90.0,
    "film_aspect_ratio": 1.5,
    "focal_length_mm": 60.0,
    "near_z": 0.1,
    "far_z": 1000.0
})";
    std::vector<std::string> required = {
        "horizontal_fov",
        "film_aspect_ratio",
        "focal_length_mm",
        "near_z",
        "far_z"
    };

    cpt::CameraLoader loader;
    for(const auto& req: required) {
        auto jobj = nlohmann::json::parse(test_string);
        jobj.erase(jobj.find(req));
        EXPECT_THROW(loader.load_camera_from_json(jobj), std::runtime_error);
    }

}

TEST(CameraLoader,TestLoadPinholePerspective)
{
    std::string test_string = R"(
{ 
    "type": "pinhole_perspective",
    "horizontal_fov": 90.0,
    "film_aspect_ratio": 1.5,
    "focal_length_mm": 60.0,
    "near_z": 0.1,
    "far_z": 1000.0
})";
    auto jobj = nlohmann::json::parse(test_string);
    cpt::CameraLoader loader;
    auto cam = loader.load_camera_from_json(jobj);
    auto pinhole = std::dynamic_pointer_cast<cpt::PinholePerspectiveCamera>(cam);
    EXPECT_TRUE(pinhole != nullptr);
    EXPECT_NEAR(pinhole->horizontal_fov().degrees(), 90.f, epsilon);
    EXPECT_NEAR(pinhole->film_aspect_ratio(), 1.5f, epsilon);
    EXPECT_NEAR(pinhole->focal_length().millimeters(), 60.0f, epsilon);
    EXPECT_NEAR(pinhole->near_z().meters(), 0.1f, epsilon);
    EXPECT_NEAR(pinhole->far_z().meters(), 1000.f, epsilon);
}

CREATE_GENERIC_TEST_MAIN
