#include <scene/geometry/vertex_container.h>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE VertexContainerTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestConstructor)
{
    // What you pass in is what you get.
    {
        cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
        BOOST_CHECK((container.positions - ref_positions).isZero());
        BOOST_CHECK((container.uvs - ref_uvs).isZero());
        BOOST_CHECK((container.normals - ref_normals).isZero());
    }

    // Pass in nothing, get nothing back.
    {
        cpt::VertexContainer container;
        BOOST_CHECK_EQUAL(container.positions.size(), 0);
        BOOST_CHECK_EQUAL(container.uvs.size(), 0);
        BOOST_CHECK_EQUAL(container.normals.size(), 0);
    }
}

BOOST_AUTO_TEST_CASE(TestNumPositions)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    BOOST_CHECK_EQUAL(container.num_positions(), 4);
}

BOOST_AUTO_TEST_CASE(TestNumUVs)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    BOOST_CHECK_EQUAL(container.num_uvs(), 4);
}

BOOST_AUTO_TEST_CASE(TestNumNormals)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    BOOST_CHECK_EQUAL(container.num_normals(), 4);
}
