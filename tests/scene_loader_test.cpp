#include <json/json.hpp>
#include <scene/loader/scene_loader.h>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SceneLoaderTest
#include <boost/test/unit_test.hpp>

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

BOOST_AUTO_TEST_CASE(TestLoadSceneFromJson)
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
