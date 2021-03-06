#include "test_common.h"

#include <scene/geometry/geometry_aggregate.h>
#include <scene/loader/mesh_loader.h>
#include <scene/geometry/triangle.h>

Eigen::Array3Xf ref_positions = [](){
    Eigen::Array3Xf tmp;
    tmp.resize(Eigen::NoChange, 4);
    tmp  << 
        -0.5, 0.5, -0.5, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.5, 0.5, -0.5, -0.5;
    return tmp;
}();

Eigen::Array2Xf ref_uvs = [](){
    Eigen::Array2Xf tmp;
    tmp.resize(Eigen::NoChange, 4);
    tmp << 
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0;
    return tmp;
}();

Eigen::Array3Xf ref_normals = [](){
    Eigen::Array3Xf tmp;
    tmp.resize(Eigen::NoChange, 4);
    tmp << 
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0;
    return tmp;
}();

void check_test_plane(const cpt::GeometryPtr& geom) {
    cpt::GeometryAggregate* agg = dynamic_cast<cpt::GeometryAggregate*>(geom.get());
    EXPECT_TRUE(agg != nullptr);

    // 2 triangles.
    EXPECT_EQ(agg->num_children(), 2);

    const cpt::GeometryPtr& g1 = agg->get_geometry(0);
    const cpt::Triangle* t1 = dynamic_cast<cpt::Triangle*>(g1.get());
    EXPECT_TRUE(t1 != nullptr);
    EXPECT_TRUE(t1->get_vertex_indices().isApprox(Eigen::Vector3i(0,1,2)));
    EXPECT_TRUE(t1->get_uv_indices().isApprox(Eigen::Vector3i(0,1,2)));
    EXPECT_TRUE(t1->get_normal_indices().isApprox(Eigen::Vector3i(0,1,2)));

    const cpt::GeometryPtr& g2 = agg->get_geometry(1);
    const cpt::Triangle* t2 = dynamic_cast<cpt::Triangle*>(g2.get());
    EXPECT_TRUE(t2 != nullptr);
    EXPECT_TRUE(t2->get_vertex_indices().isApprox(Eigen::Vector3i(2,1,3)));
    EXPECT_TRUE(t2->get_uv_indices().isApprox(Eigen::Vector3i(2,1,3)));
    EXPECT_TRUE(t2->get_normal_indices().isApprox(Eigen::Vector3i(2,1,3)));

    // Two vertex containers should point to the same.
    const cpt::VertexContainerPtr& v1 = t1->get_vertex_container();
    const cpt::VertexContainerPtr& v2 = t2->get_vertex_container();
    EXPECT_EQ(v1.get(), v2.get());
    EXPECT_EQ(v1->num_positions(), 4);
    EXPECT_EQ(v1->num_uvs(), 4);
    EXPECT_EQ(v1->num_normals(), 4);

    EXPECT_TRUE((v1->positions - ref_positions).isZero());
    EXPECT_TRUE((v1->uvs - ref_uvs).isZero());
    EXPECT_TRUE((v1->normals - ref_normals).isZero());

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE((v1->position(i) - ref_positions.col(i).matrix()).isZero());
        EXPECT_TRUE((v1->uv(i) - ref_uvs.col(i).matrix()).isZero());
        EXPECT_TRUE((v1->normal(i) - ref_normals.col(i).matrix()).isZero());
    }
}

cpt::GeometryPtr construct_test_plane() {
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
    return builder.construct();
}

void check_pinhole_perspective_camera_equal(
    const cpt::PinholePerspectiveCameraPtr& cam1,
    const cpt::PinholePerspectiveCameraPtr& cam2,
    double epsilon) {
    EXPECT_TRUE(cam1 != nullptr);
    EXPECT_TRUE(cam2 != nullptr);

    EXPECT_NEAR(cam1->horizontal_fov().degrees(), cam2->horizontal_fov().degrees(), epsilon);
    EXPECT_NEAR(cam1->film_aspect_ratio(), cam2->film_aspect_ratio(), epsilon);
    EXPECT_NEAR(cam1->focal_length().meters(), cam2->focal_length().meters(), epsilon);
    EXPECT_NEAR(cam1->near_z().meters(), cam2->near_z().meters(), epsilon);
    EXPECT_NEAR(cam1->far_z().meters(), cam2->far_z().meters(), epsilon);
}

void check_transforms_equal(
    const cpt::Transform& xform1,
    const cpt::Transform& xform2,
    double epsilon) {
    EXPECT_TRUE((xform1.translation() - xform2.translation()).isZero(epsilon));
    EXPECT_TRUE((xform1.rotation() * xform1.scale().asDiagonal() 
        - xform2.rotation() * xform2.scale().asDiagonal()).isZero(epsilon));
}
