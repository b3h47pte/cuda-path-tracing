#include <math/angles.h>
#include <math/rotation.h>
#include <scene/loader/xform_loader.h>
#include <json/json.hpp>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE XformLoaderTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestLoadEmpty)
{
    std::string test_string = R"(
{ 
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    BOOST_CHECK(xform.translation().isZero());
    BOOST_CHECK(xform.scale().isOnes());
    BOOST_CHECK(xform.rotation().isIdentity());
}

BOOST_AUTO_TEST_CASE(TestLoadTranslation)
{
    std::string test_string = R"(
{ 
    "translation": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    BOOST_CHECK(xform.translation().isApprox(Eigen::Vector3f(3.f, -2.f, 1.5f)));
    BOOST_CHECK(xform.scale().isOnes());
    BOOST_CHECK(xform.rotation().isIdentity());
}

BOOST_AUTO_TEST_CASE(TestLoadScale)
{
    std::string test_string = R"(
{ 
    "scale": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    BOOST_CHECK(xform.translation().isZero());
    BOOST_CHECK(xform.scale().isApprox(Eigen::Vector3f(3.f, -2.f, 1.5f)));
    BOOST_CHECK(xform.rotation().isIdentity());
}

BOOST_AUTO_TEST_CASE(TestLoadRotationNoType)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5]
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(loader.load_xform_from_json(jobj), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadRotationUnsupportedType)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5],
    "rotation_type": "_THIS_IS_NOT_SUPPORTED_"
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    BOOST_CHECK_THROW(loader.load_xform_from_json(jobj), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestLoadRotationEulerXYZ)
{
    std::string test_string = R"(
{ 
    "rotation": [3.0, -2.0, 1.5],
    "rotation_type": "euler_xyz"
})";
    cpt::XformLoader loader;
    auto jobj = nlohmann::json::parse(test_string);
    auto xform = loader.load_xform_from_json(jobj);
    BOOST_CHECK(xform.rotation().isApprox(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians()))));
}

BOOST_AUTO_TEST_CASE(TestLoadAll)
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
    BOOST_CHECK(xform.translation().isApprox(Eigen::Vector3f(1.f, -2.2f, 1.1f)));
    BOOST_CHECK(xform.scale().isApprox(Eigen::Vector3f(0.2f, -1.f, 5.5f)));
    BOOST_CHECK(xform.rotation().isApprox(cpt::get_euler_xyz_rotation_matrix(
        Eigen::Vector3f(
            cpt::Angle::from_degrees(3.f).radians(),
            cpt::Angle::from_degrees(-2.f).radians(),
            cpt::Angle::from_degrees(1.5f).radians()))));

}
