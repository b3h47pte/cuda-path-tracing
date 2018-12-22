#include <scene/camera/pinhole_perspective_camera.h>
#include <scene/loader/camera_loader.h>
#include <json/json.hpp>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CameraLoaderTest
#include <boost/test/unit_test.hpp>

namespace {
constexpr double epsilon = 1e-6;
}

BOOST_AUTO_TEST_CASE(TestLoadNoType)
{
    std::string test_string = R"(
{ 
})";

    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(cpt::load_camera_from_json(jobj), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadUnsupportedType)
{
    std::string test_string = R"(
{ 
    "type": "_THIS_TYPE_NOT_SUPPORTED_"
})";

    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(cpt::load_camera_from_json(jobj), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadIncompletePinholePerspective)
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

    for(const auto& req: required) {
        auto jobj = nlohmann::json::parse(test_string);
        jobj.erase(jobj.find(req));
        BOOST_CHECK_THROW(cpt::load_camera_from_json(jobj), std::runtime_error);
    }

}

BOOST_AUTO_TEST_CASE(TestLoadPinholePerspective)
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
    auto cam = cpt::load_camera_from_json(jobj);
    auto pinhole = std::dynamic_pointer_cast<cpt::PinholePerspectiveCamera>(cam);
    BOOST_CHECK(pinhole != nullptr);
    BOOST_CHECK_CLOSE(pinhole->horizontal_fov().degrees(), 90.f, epsilon);
    BOOST_CHECK_CLOSE(pinhole->film_aspect_ratio(), 1.5f, epsilon);
    BOOST_CHECK_CLOSE(pinhole->focal_length().millimeters(), 60.0f, epsilon);
    BOOST_CHECK_CLOSE(pinhole->near_z().meters(), 0.1f, epsilon);
    BOOST_CHECK_CLOSE(pinhole->far_z().meters(), 1000.f, epsilon);
}
