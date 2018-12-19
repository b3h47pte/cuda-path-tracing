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
    // test_scene.json - Ensure that default loading of a single
    // mesh works and that the root object is successfully ignored.
    {
        auto scene = cpt::load_scene_from_json("data/test_scene.json");
        BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
        check_test_plane(scene->geometry(0));
    }

    // test_scene2.json - Ensure that the root object is not actually needed.
    {
        auto scene = cpt::load_scene_from_json("data/test_scene2.json");
        BOOST_CHECK_EQUAL(scene->num_geometry(), 1);
        check_test_plane(scene->geometry(0));
    }

    // test_scene3.json - Check to make sure an object of unknown type throws an error.
    BOOST_CHECK_THROW(cpt::load_scene_from_json("data/test_scene3.json"), std::runtime_error);
    // test_scene4.json - Typeless entry should throw an error.
    BOOST_CHECK_THROW(cpt::load_scene_from_json("data/test_scene4.json"), std::runtime_error);

    // test_scene5.json - Ensure that nested children works.
    {
        auto scene = cpt::load_scene_from_json("data/test_scene5.json");
        BOOST_CHECK_EQUAL(scene->num_geometry(), 3);
        check_test_plane(scene->geometry(0));
        check_test_plane(scene->geometry(1));
        check_test_plane(scene->geometry(2));
    }

    // test_scene6.json - Ensure that a "mesh" object requires a filename.
    BOOST_CHECK_THROW(cpt::load_scene_from_json("data/test_scene6.json"), std::runtime_error);
}
