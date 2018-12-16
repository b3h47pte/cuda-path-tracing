#include <scene/loader/mesh_loader.h>
#include <scene/geometry/geometry_aggregate.h>
#include <scene/geometry/triangle.h>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MeshLoaderTest
#include <boost/test/unit_test.hpp>

namespace {

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

}

void check_test_plane(const cpt::GeometryPtr& geom) {
    cpt::GeometryAggregate* agg = dynamic_cast<cpt::GeometryAggregate*>(geom.get());
    BOOST_CHECK(agg != nullptr);

    // 2 triangles.
    BOOST_CHECK_EQUAL(agg->num_children(), 2);

    const cpt::GeometryPtr& g1 = agg->get_geometry(0);
    const cpt::Triangle* t1 = dynamic_cast<cpt::Triangle*>(g1.get());
    BOOST_CHECK(t1 != nullptr);
    BOOST_CHECK(t1->get_vertex_indices().isApprox(Eigen::Vector3i(0,1,2)));
    BOOST_CHECK(t1->get_uv_indices().isApprox(Eigen::Vector3i(0,1,2)));
    BOOST_CHECK(t1->get_normal_indices().isApprox(Eigen::Vector3i(0,1,2)));

    const cpt::GeometryPtr& g2 = agg->get_geometry(1);
    const cpt::Triangle* t2 = dynamic_cast<cpt::Triangle*>(g2.get());
    BOOST_CHECK(t2 != nullptr);
    BOOST_CHECK(t2->get_vertex_indices().isApprox(Eigen::Vector3i(2,1,3)));
    BOOST_CHECK(t2->get_uv_indices().isApprox(Eigen::Vector3i(2,1,3)));
    BOOST_CHECK(t2->get_normal_indices().isApprox(Eigen::Vector3i(2,1,3)));

    // Two vertex containers should point to the same.
    const cpt::VertexContainerPtr& v1 = t1->get_vertex_container();
    const cpt::VertexContainerPtr& v2 = t2->get_vertex_container();
    BOOST_CHECK_EQUAL(v1.get(), v2.get());
    BOOST_CHECK_EQUAL(v1->num_positions(), 4);
    BOOST_CHECK_EQUAL(v1->num_uvs(), 4);
    BOOST_CHECK_EQUAL(v1->num_normals(), 4);

    BOOST_CHECK((v1->positions - ref_positions).isZero());
    BOOST_CHECK((v1->uvs - ref_uvs).isZero());
    BOOST_CHECK((v1->normals - ref_normals).isZero());

    for (int i = 0; i < 4; ++i) {
        BOOST_CHECK((v1->position(i) - ref_positions.col(i).matrix()).isZero());
        BOOST_CHECK((v1->uv(i) - ref_uvs.col(i).matrix()).isZero());
        BOOST_CHECK((v1->normal(i) - ref_normals.col(i).matrix()).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestLoadMeshDispatch) {
    // File must exist.
    BOOST_CHECK_THROW(cpt::load_mesh_from_file("data/_does_not_exist.obj"), std::runtime_error);

    // We must support the file type.
    BOOST_CHECK_THROW(cpt::load_mesh_from_file("data/support.abc"), std::runtime_error);
    BOOST_CHECK_THROW(cpt::load_mesh_from_file("data/support.fbx"), std::runtime_error);
    BOOST_CHECK_THROW(cpt::load_mesh_from_file("data/support.ply"), std::runtime_error);

    // We must be robust to capitilization for extension detection.
    // Check that we can successfully load/dispatch in the presence of non-lowercase.
    check_test_plane(cpt::load_mesh_from_file("data/plane.obj"));
    check_test_plane(cpt::load_mesh_from_file("data/plane.OBJ"));
}

BOOST_AUTO_TEST_CASE(TestLoadObj) {
    // Ensure that calling the load obj function directly also succeeds.
    check_test_plane(cpt::load_obj_from_file("data/plane.obj"));
}

BOOST_AUTO_TEST_CASE(TestGeometryAggregateBuilder) {
    // Test empty.
    {
        cpt::GeometryAggregateBuilder builder;
        auto geom = builder.construct();
        cpt::GeometryAggregate* agg = dynamic_cast<cpt::GeometryAggregate*>(geom.get());
        BOOST_CHECK(agg != nullptr);
        BOOST_CHECK_EQUAL(agg->num_children(), 0);
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
        BOOST_CHECK_THROW(builder.construct(), std::runtime_error);
    }
}
