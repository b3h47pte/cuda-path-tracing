#include <gtest/gtest.h>
#include <json/json.hpp>
#include <scene/loader/camera_loader.h>
#include <scene/loader/scene_loader.h>
#include <math/rotation.h>
#include "mocks/camera_loader_mock.h"
#include "mocks/mesh_loader_mock.h"
#include "mocks/scene_builder_mock.h"
#include "mocks/scene_loader_mock.h"
#include "mocks/xform_loader_mock.h"
#include "test_common.h"

using ::testing::AtLeast;
using ::testing::InSequence;

namespace {
constexpr double epsilon = 1e-6;

cpt::SceneLoader::SceneLoaderDependencies create_full_mock_dependencies() {
    cpt::SceneLoader::SceneLoaderDependencies deps;
    deps.camera_loader = std::make_shared<MockCameraLoader>();
    deps.mesh_loader = std::make_shared<MockMeshLoader>();
    deps.scene_builder = std::make_shared<MockSceneBuilder>();
    deps.xform_loader = std::make_shared<MockXformLoader>();
    return deps;
}

}

TEST(SceneBuilder,AddGeometry)
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
        EXPECT_EQ(scene->num_geometry(), 2);
        EXPECT_EQ(scene->geometry(0).get(), p1.get());
        EXPECT_EQ(scene->geometry(1).get(), p2.get());
    }
}

TEST(SceneBuilder,AddCamera)
{
    auto cam = std::make_shared<TestCamera>();
    cpt::SceneBuilder builder;
    builder.add_camera("cam0", cam);
    // Can't add camera with the same name.
    EXPECT_THROW(builder.add_camera("cam0", cam), std::runtime_error);

    auto scene = builder.construct();
    EXPECT_EQ(scene->num_cameras(), 1);
    EXPECT_TRUE(scene->has_camera("cam0"));
    EXPECT_EQ(scene->camera("cam0").get(), cam.get());
}

TEST(SceneBuilder,PushPopTransforms)
{
    cpt::SceneBuilder builder;
    check_transforms_equal(builder.current_transform(), cpt::Transform(), epsilon);

    cpt::Transform ref_xform;
    ref_xform.set_translation(Eigen::Vector3f(1.f, -2.2f, 1.1f));
    ref_xform.set_scale(Eigen::Vector3f(0.2f, -1.f, 5.5f));
    ref_xform.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians())));
    builder.push_transform(ref_xform);
    check_transforms_equal(builder.current_transform(), ref_xform, epsilon);
    builder.pop_transform();
    check_transforms_equal(builder.current_transform(), cpt::Transform(), epsilon);
    builder.push_transform(ref_xform);
    check_transforms_equal(builder.current_transform(), ref_xform, epsilon);

    cpt::Transform ref_xform2;
    ref_xform2.set_translation(Eigen::Vector3f(-1.f, 2.2f, -1.1f));
    ref_xform2.set_scale(Eigen::Vector3f::Constant(0.5f));
    ref_xform2.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(1.f).radians(),
            cpt::Angle::from_degrees(2.f).radians(),
            cpt::Angle::from_degrees(3.f).radians())));
    builder.push_transform(ref_xform2);
    check_transforms_equal(builder.current_transform(), ref_xform * ref_xform2, epsilon);
}

TEST(SceneBuilder,AddGeometryXform)
{
    cpt::SceneBuilder builder;

    cpt::Transform ref_xform;
    ref_xform.set_translation(Eigen::Vector3f(1.f, -2.2f, 1.1f));
    ref_xform.set_scale(Eigen::Vector3f(0.2f, -1.f, 5.5f));
    ref_xform.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians())));
    builder.push_transform(ref_xform);

    auto cam = std::make_shared<TestCamera>();
    builder.add_camera("cam0", cam);
    
    auto scene = builder.construct();
    check_transforms_equal(scene->camera("cam0")->object_to_world_xform(), ref_xform, epsilon);
}

TEST(SceneBuilder,AddCameraXform)
{
    cpt::SceneBuilder builder;

    cpt::Transform ref_xform;
    ref_xform.set_translation(Eigen::Vector3f(1.f, -2.2f, 1.1f));
    ref_xform.set_scale(Eigen::Vector3f(0.2f, -1.f, 5.5f));
    ref_xform.set_rotation(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians())));
    builder.push_transform(ref_xform);

    auto p1 = construct_test_plane();
    builder.add_geometry(p1);
    
    auto scene = builder.construct();
    check_transforms_equal(scene->geometry(0)->object_to_world_xform(), ref_xform, epsilon);

}

TEST(SceneLoader,LoadSceneFromJsonIgnoreRoot)
{
    // Ensure that default loading of a single
    // mesh works and that the root object is successfully ignored.
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
    
    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);
    auto jobj = nlohmann::json::parse(test_string);
    
    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1);
    }

    loader.load_scene_from_json(jobj, "data/");
}

TEST(SceneLoader,TestLoadSceneFromJsonNoRoot)
{
    // Ensure that the root object is not actually needed.
    std::string test_string = R"(
{
"type": "mesh",
"filename": "plane.obj"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);

    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1);
    }

    loader.load_scene_from_json(jobj, "data/");

}

TEST(SceneLoader,TestLoadSceneFromJsonUnknownType)
{
    // Check to make sure an object of unknown type throws an error.
    std::string test_string = R"(
{
    "type": "THIS_SHOULD_THROW_AN_ERROR"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneFromJsonTypeless)
{
    // Typeless entry should throw an error.
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
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneFromJsonNestedGeometry)
{
    // Ensure that nested children works.
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
    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);
    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1).RetiresOnSaturation();
    }

    loader.load_scene_from_json(jobj, "data/");
}

TEST(SceneLoader,TestLoadSceneFromJsonMeshFilename)
{
    // Ensure that a "mesh" object requires a filename.
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
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneFromJsonCameraNoId)
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
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneFromJsonCameraNoObject)
{
    std::string test_string = R"(
{
    "type": "camera",
    "id": "cam0"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneFromJsonCamera)
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
    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);
    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockCameraLoader&>(*deps.camera_loader),
            load_camera_from_json).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_camera).Times(1);
    }
    loader.load_scene_from_json(jobj, "data/");
}

TEST(SceneLoader,TestLoadSceneFromJsonXformNoObject)
{
    std::string test_string = R"(
{
    "type": "xform"
}
)";
    auto jobj = nlohmann::json::parse(test_string);
    cpt::SceneLoader loader;
    EXPECT_THROW(loader.load_scene_from_json(jobj, "data/"), std::runtime_error);
}

TEST(SceneLoader,TestLoadSceneXformIntoGeometry)
{
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

    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);

    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockXformLoader&>(*deps.xform_loader),
            load_xform_from_json).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            push_transform).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1);
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            pop_transform).Times(1);
    }

    loader.load_scene_from_json(jobj, "data/");
}

TEST(SceneLoader,TestLoadSceneXformNestedIntoGeometry)
{
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
    auto deps = create_full_mock_dependencies();
    cpt::SceneLoader loader(deps);

    {
        InSequence tmp;
        EXPECT_CALL(
            dynamic_cast<MockXformLoader&>(*deps.xform_loader),
            load_xform_from_json).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            push_transform).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockXformLoader&>(*deps.xform_loader),
            load_xform_from_json).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            push_transform).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockMeshLoader&>(*deps.mesh_loader),
            load_mesh_from_file).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            add_geometry).Times(1).RetiresOnSaturation();
        EXPECT_CALL(
            dynamic_cast<MockSceneBuilder&>(*deps.scene_builder),
            pop_transform).Times(2).RetiresOnSaturation();
    }

    loader.load_scene_from_json(jobj, "data/");
}

CREATE_GENERIC_TEST_MAIN
