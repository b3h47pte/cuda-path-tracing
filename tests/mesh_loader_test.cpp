#include <scene/loader/mesh_loader.h>
#include <scene/geometry/geometry_aggregate.h>
#include <scene/geometry/triangle.h>
#include "test_common.h"

namespace {

}

TEST(MeshLoader,TestLoadMeshDispatch) {
    cpt::MeshLoader loader;

    // File must exist.
    EXPECT_THROW(loader.load_mesh_from_file("data/_does_not_exist.obj"), std::runtime_error);

    // We must support the file type.
    EXPECT_THROW(loader.load_mesh_from_file("data/support.abc"), std::runtime_error);
    EXPECT_THROW(loader.load_mesh_from_file("data/support.fbx"), std::runtime_error);
    EXPECT_THROW(loader.load_mesh_from_file("data/support.ply"), std::runtime_error);

    // We must be robust to capitilization for extension detection.
    // Check that we can successfully load/dispatch in the presence of non-lowercase.
    check_test_plane(loader.load_mesh_from_file("data/plane.obj"));
    check_test_plane(loader.load_mesh_from_file("data/plane.OBJ"));
}

TEST(MeshLoader,TestLoadObj) {
    cpt::MeshLoader loader;
    // Ensure that calling the load obj function directly also succeeds.
    check_test_plane(loader.load_obj_from_file("data/plane.obj"));
}

TEST(MeshLoader,TestGeometryAggregateBuilder) {
    // Test empty.
    {
        cpt::GeometryAggregateBuilder builder;
        auto geom = builder.construct();
        cpt::GeometryAggregate* agg = dynamic_cast<cpt::GeometryAggregate*>(geom.get());
        EXPECT_TRUE(agg != nullptr);
        EXPECT_EQ(agg->num_children(), 0);
    }

    // Test creating the standard plane.
    {
        cpt::GeometryAggregateBuilder builder;
        for (auto i = 0; i < 4; ++i) {
            builder.add_vertex_position(ref_positions.col(i));
            builder.add_vertex_uv(ref_uvs.col(i));
            builder.add_vertex_normal(ref_normals.col(i));
        }
        builder.add_face(
            Eigen::Vector3i(0, 1, 2),
            Eigen::Vector3i(0, 1, 2),
            Eigen::Vector3i(0, 1, 2));

        builder.add_face(
            Eigen::Vector3i(2, 1, 3),
            Eigen::Vector3i(2, 1, 3),
            Eigen::Vector3i(2, 1, 3));
        check_test_plane(builder.construct());
    }

    // Test face out of bound of vertex container.
    {
        cpt::GeometryAggregateBuilder builder;
        builder.add_face(
            Eigen::Vector3i(0, 1, 2),
            Eigen::Vector3i(0, 1, 2),
            Eigen::Vector3i(0, 1, 2));
        EXPECT_THROW(builder.construct(), std::runtime_error);
    }
}

CREATE_GENERIC_TEST_MAIN
