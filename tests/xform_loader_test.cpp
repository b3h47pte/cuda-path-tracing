#include <math/angles.h>
#include <math/rotation.h>
#include <scene/loader/xform_loader.h>
#include <json/json.hpp>
#include "test_common.h"

TEST(XformLoader,TestLoadEmpty)
{
    std::string test_string = R"(
{ 
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    EXPECT_TRUE(xform.translation().isZero());
    EXPECT_TRUE(xform.scale().isOnes());
    EXPECT_TRUE(xform.rotation().isIdentity());
}

TEST(XformLoader,TestLoadTranslation)
{
    std::string test_string = R"(
{ 
    "translation": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    EXPECT_TRUE(xform.translation().isApprox(Eigen::Vector3f(3.f, -2.f, 1.5f)));
    EXPECT_TRUE(xform.scale().isOnes());
    EXPECT_TRUE(xform.rotation().isIdentity());
}

TEST(XformLoader,TestLoadScale)
{
    std::string test_string = R"(
{ 
    "scale": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    EXPECT_TRUE(xform.translation().isZero());
    EXPECT_TRUE(xform.scale().isApprox(Eigen::Vector3f(3.f, -2.f, 1.5f)));
    EXPECT_TRUE(xform.rotation().isIdentity());
}

TEST(XformLoader,TestLoadRotationNoType)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    EXPECT_THROW(loader.load_xform_from_json(jobj), std::runtime_error);
}

TEST(XformLoader,TestLoadRotationUnsupportedType)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5],
    "rotation_type": "_THIS_IS_NOT_SUPPORTED_"
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    EXPECT_THROW(loader.load_xform_from_json(jobj), std::runtime_error);
}

TEST(XformLoader,TestLoadRotationEulerXYZ)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5],
    "rotation_type": "euler_xyz"
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    EXPECT_TRUE(xform.rotation().isApprox(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians()))));
}

TEST(XformLoader,TestLoadAll)
{
    std::string test_string = R"(
{ 
    "translation": [1.0, -2.2, 1.1],
    "scale": [0.2, -1.0, 5.5],
    "rotation": [3.0, -2.0, 1.5],
    "rotation_type": "euler_xyz"
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    EXPECT_TRUE(xform.translation().isApprox(Eigen::Vector3f(1.f, -2.2f, 1.1f)));
    EXPECT_TRUE(xform.scale().isApprox(Eigen::Vector3f(0.2f, -1.f, 5.5f)));
    EXPECT_TRUE(xform.rotation().isApprox(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians()))));

}

CREATE_GENERIC_TEST_MAIN
