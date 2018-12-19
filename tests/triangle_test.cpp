#include <scene/geometry/triangle.h>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TriangleTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestConstructor)
{
    // What you pass in is what you get.
    {
        auto container = std::make_shared<cpt::VertexContainer>(ref_positions, ref_uvs, ref_normals);
        Eigen::Vector3i position_idx(0, 1, 2);
        Eigen::Vector3i uv_idx(2, 1, 0);
        Eigen::Vector3i normal_idx(1, 2, 0);
        cpt::Triangle tri(container, position_idx, uv_idx, normal_idx);
        BOOST_CHECK(tri.get_vertex_indices().isApprox(position_idx));
        BOOST_CHECK(tri.get_uv_indices().isApprox(uv_idx));
        BOOST_CHECK(tri.get_normal_indices().isApprox(normal_idx));
    }
}
