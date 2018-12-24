#include <json/json.hpp>
#include <scene/loader/camera_loader.h>
#include <scene/loader/scene_loader.h>
#include <math/rotation.h>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SceneLoaderTest
#include <boost/test/unit_test.hpp>

namespace {
constexpr double epsilon = 1e-6;
}

BOOST_AUTO_TEST_CASE(TestSceneBuilder)
{
    // Construct reference geometry.
    auto p1 = construct_test_plane();
    auto p2 = construct_test_plane();

    // Test geometry addition.
    {
        cpt::SceneBuilder builder;
        builder.add_geometry(p1);
        builder.add_geometry(p2);

        auto scene = builder.construct();
        BOOST_CHECK_EQUAL(scene->num_geometry(), 2);
        BOOST_CHECK_EQUAL(scene->geometry(0).get(), p1.get());
        BOOST_CHECK_EQUAL(scene->geometry(1).get(), p2.get());
    }
}

// TODO: Somehow split the testing of the parsing of the scene JSON from
//       the testing of the separate load_*_from_json functions and from
//       the testing of the SceneBuilder. Will probably need to restructure
//       this API to use dependency injection.

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonIgnoreRoot)
{
    // Ensure that default loading of a single
    // mesh works and that the root object is successfully ignored.
    {
        std::string test_string = R"(
{
    "type": "root",
    "children": [
        {
            "type": "mesh",
            "filename": "plane.obj"
        }
    ]
})";
        auto jobj = nlohmann::json::parse(test_string);
        auto scene = cpt::load_scene_from_json(jobj, "data/");
        BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
        check_test_plane(scene->geometry(0));
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonNoRoot)
{
    // Ensure that the root object is not actually needed.
    {
        std::string test_string = R"(
{
    "type": "mesh",
    "filename": "plane.obj"
}
)";
        auto jobj = nlohmann::json::parse(test_string);
        auto scene = cpt::load_scene_from_json(jobj, "data/");
        BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
        check_test_plane(scene->geometry(0));
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonUnknownType)
{
    // Check to make sure an object of unknown type throws an error.
    {
        std::string test_string = R"(
{
    "type": "THIS_SHOULD_THROW_AN_ERROR"
}
)";
        auto jobj = nlohmann::json::parse(test_string);
        BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonTypeless)
{
    // Typeless entry should throw an error.
    {
        std::string test_string = R"(
{
    "type": "root",
    "children": [
        {
            "filename": "plane.obj"
        }
    ]
}
)";
        auto jobj = nlohmann::json::parse(test_string);
        BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonNestedGeometry)
{
    // Ensure that nested children works.
    {
        std::string test_string = R"(
{
    "type": "root",
    "children": [
        {
            "type": "mesh",
            "filename": "plane.obj"
        },
        {
            "type": "mesh",
            "filename": "plane.obj",
            "children": [
                {
                    "type": "mesh",
                    "filename": "plane.obj"
                }
            ]
        }
    ]
}
)";
        auto jobj = nlohmann::json::parse(test_string);
        auto scene = cpt::load_scene_from_json(jobj, "data/");

        BOOST_CHECK_EQUAL(scene->num_geometry(), 3);
        check_test_plane(scene->geometry(0));
        check_test_plane(scene->geometry(1));
        check_test_plane(scene->geometry(2));
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonMeshFilename)
{
    // Ensure that a "mesh" object requires a filename.
    {
        std::string test_string = R"(
{
    "type": "root",
    "children": [
        {
            "type": "mesh"
        }
    ]
}
)";
        auto jobj = nlohmann::json::parse(test_string);
        BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
    }
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonCameraNoId)
{
    std::string test_string = R"(
{
    "type": "camera",
    "object": {
        "type": "pinhole_perspective",
        "horizontal_fov": 90.0,
        "film_aspect_ratio": 1.5,
        "focal_length_mm": 60.0,
        "near_z": 0.1,
        "far_z": 1000.0
    }
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonCameraNoObject)
{
    std::string test_string = R"(
{
    "type": "camera",
    "id": "cam0"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonCamera)
{
    std::string test_string = R"(
{
    "type": "camera",
    "id": "cam0",
    "object": {
        "type": "pinhole_perspective",
        "horizontal_fov": 90.0,
        "film_aspect_ratio": 1.5,
        "focal_length_mm": 60.0,
        "near_z": 0.1,
        "far_z": 1000.0
    }
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    auto scene = cpt::load_scene_from_json(jobj, "data/");
    auto camera = cpt::load_camera_from_json(jobj["object"]);

    // Ideally I would do some dependency injection to mock
    // cpt::load_camera_from_json and just check if it's called.
    BOOST_CHECK(scene->has_camera("cam0"));
    BOOST_CHECK_EQUAL(scene->num_cameras(), 1);
    auto test_camera = scene->camera("cam0");
    check_pinhole_perspective_camera_equal(
        std::dynamic_pointer_cast<cpt::PinholePerspectiveCamera>(camera),
        std::dynamic_pointer_cast<cpt::PinholePerspectiveCamera>(test_camera),
        epsilon);
}

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJsonXformNoObject)
{
    std::string test_string = R"(
{
    "type": "xform"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(cpt::load_scene_from_json(jobj, "data/"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadSceneXformIntoGeometry)
{
    cpt::Transform ref_xform;
    ref_xform.set_translation(Eigen::Vector3f(1.f, -2.2f, 1.1f));
    ref_xform.set_scale(Eigen::Vector3f(0.2f, -1.f, 5.5f));
    ref_xform.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians())));
    std::string test_string = R"(
{
    "type": "xform",
    "object": {
        "translation": [1.0, -2.2, 1.1],
        "scale": [0.2, -1.0, 5.5],
        "rotation": [3.0, -2.0, 1.5],
        "rotation_type": "euler_xyz"
    },
    "children": [
        {
            "type": "mesh",
            "filename": "plane.obj"
        }
    ]
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    auto scene = cpt::load_scene_from_json(jobj, "data/");
    BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
    check_transforms_equal(scene->geometry(0)->object_to_world_xform(), ref_xform, epsilon);
}

BOOST_AUTO_TEST_CASE(TestLoadSceneXformNestedIntoGeometry)
{
    cpt::Transform ref_xform;
    ref_xform.set_translation(Eigen::Vector3f(1.f, -2.2f, 1.1f));
    ref_xform.set_scale(Eigen::Vector3f(0.2f, -1.f, 5.5f));
    ref_xform.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians())));

    cpt::Transform ref_xform2;
    ref_xform2.set_translation(Eigen::Vector3f(-1.f, 2.2f, -1.1f));
    ref_xform2.set_scale(Eigen::Vector3f::Constant(0.5f));
    ref_xform2.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(1.f).radians(),
            cpt::Angle::from_degrees(2.f).radians(),
            cpt::Angle::from_degrees(3.f).radians())));

    std::string test_string = R"(
{
    "type": "xform",
    "object": {
        "translation": [1.0, -2.2, 1.1],
        "scale": [0.2, -1.0, 5.5],
        "rotation": [3.0, -2.0, 1.5],
        "rotation_type": "euler_xyz"
    },
    "children": [
        {
            "type": "xform",
            "object": {
                "translation": [-1.0, 2.2, -1.1],
                "scale": [0.5, 0.5, 0.5],
                "rotation": [1.0, 2.0, 3.0],
                "rotation_type": "euler_xyz"
            },
            "children": [
                {
                    "type": "mesh",
                    "filename": "plane.obj"
                }
            ]
        }
    ]
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    auto scene = cpt::load_scene_from_json(jobj, "data/");
    BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
    check_transforms_equal(scene->geometry(0)->object_to_world_xform(), ref_xform * ref_xform2, epsilon);
}
