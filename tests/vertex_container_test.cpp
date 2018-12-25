#include <scene/geometry/vertex_container.h>
#include "test_common.h"

TEST(VertexContainer,TestConstructor)
{
    // What you pass in is what you get.
    {
        cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
        EXPECT_TRUE((container.positions - ref_positions).isZero());
        EXPECT_TRUE((container.uvs - ref_uvs).isZero());
        EXPECT_TRUE((container.normals - ref_normals).isZero());
    }

    // Pass in nothing, get nothing back.
    {
        cpt::VertexContainer container;
        EXPECT_EQ(container.positions.size(), 0);
        EXPECT_EQ(container.uvs.size(), 0);
        EXPECT_EQ(container.normals.size(), 0);
    }
}

TEST(VertexContainer,TestNumPositions)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    EXPECT_EQ(container.num_positions(), 4);
}

TEST(VertexContainer,TestNumUVs)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    EXPECT_EQ(container.num_uvs(), 4);
}

TEST(VertexContainer,TestNumNormals)
{
    cpt::VertexContainer container(ref_positions, ref_uvs, ref_normals);
    EXPECT_EQ(container.num_normals(), 4);
}

CREATE_GENERIC_TEST_MAIN
