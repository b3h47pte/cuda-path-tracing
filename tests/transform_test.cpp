#include <math/transform.h>
#include <Eigen/Geometry>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TransformTest
#include <boost/test/unit_test.hpp>

namespace {

constexpr double epsilon = 1e-4;
std::vector<Eigen::Vector3f> test_vectors = {
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},    
    {0.f, 1.f, 0.f},    
    {0.f, 0.f, 1.f},    
    {-1.f, 5.f, 6.f}
};
}

BOOST_AUTO_TEST_CASE(TestIdentityAndReset)
{
    cpt::Transform xform;
    BOOST_CHECK(xform.translation().isZero());
    BOOST_CHECK(xform.scale().isOnes());
    BOOST_CHECK(xform.rotation().isIdentity());

    xform.set_translation(Eigen::Vector3f::Constant(1.f));
    {
        Eigen::AngleAxisf base_ref_rotation(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());
        Eigen::Matrix3f ref_rotation = base_ref_rotation.toRotationMatrix();
        xform.set_rotation(ref_rotation);
    }
    xform.set_scale(Eigen::Vector3f::Constant(0.5f));
    xform.reset();
    BOOST_CHECK(xform.translation().isZero());
    BOOST_CHECK(xform.scale().isOnes());
    BOOST_CHECK(xform.rotation().isIdentity());

    for (const auto& test_vector : test_vectors) {
        BOOST_CHECK((xform * test_vector - test_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - test_vector).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestTranslation)
{
    const Eigen::Vector3f ref_trans(0.1f, -0.5f, 2.f);
    cpt::Transform xform;
    xform.set_translation(ref_trans);
    BOOST_CHECK(xform.translation().isApprox(ref_trans));

    {
        Eigen::Vector3f test_vector(0.f, 0.f, 0.f);
        Eigen::Vector3f ref_vector(0.1f, -0.5f, 2.f);
        BOOST_CHECK((xform * test_vector - test_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector).isZero());
    }

    {
        Eigen::Vector3f test_vector(8.f, -3.f, 0.3f);
        Eigen::Vector3f ref_vector(8.1f, -3.5f, 2.3f);
        BOOST_CHECK((xform * test_vector - test_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestRotation)
{
    Eigen::AngleAxisf base_ref_rotation(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());
    Eigen::Matrix3f ref_rotation = base_ref_rotation.toRotationMatrix();

    cpt::Transform xform;
    xform.set_rotation(ref_rotation);
    BOOST_CHECK(xform.rotation().isApprox(ref_rotation));

    for (const auto& test_vector : test_vectors) {
        const Eigen::Vector3f ref_vector = ref_rotation * test_vector;
        BOOST_CHECK((xform * test_vector - ref_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestInvalidRotation)
{
    Eigen::Matrix3f invalid_rotation;
    invalid_rotation << 
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f;

    cpt::Transform xform;
    BOOST_CHECK_THROW(xform.set_rotation(invalid_rotation), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TestScale)
{
    const Eigen::Vector3f ref_scale(0.1f, -0.5f, 2.f);
    cpt::Transform xform;
    xform.set_scale(ref_scale);
    BOOST_CHECK(xform.scale().isApprox(ref_scale));

    {
        Eigen::Vector3f test_vector(0.f, 0.f, 0.f);
        BOOST_CHECK((xform * test_vector - test_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - test_vector).isZero());
    }

    {
        Eigen::Vector3f test_vector(8.f, -3.f, 0.3f);
        Eigen::Vector3f ref_vector(0.8f, 1.5f, 0.6f);
        BOOST_CHECK((xform * test_vector - ref_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestAllApply)
{
    // This just ensures we're doing the order of operations correctly (scale, rotate, translate).
    const Eigen::Vector3f ref_trans(0.1f, -0.5f, 2.f);
    const Eigen::Vector3f ref_scale(0.5f, -0.1f, 3.f);
    Eigen::AngleAxisf base_ref_rotation(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());
    Eigen::Matrix3f ref_rotation = base_ref_rotation.toRotationMatrix();

    cpt::Transform xform;
    xform.set_translation(ref_trans);
    xform.set_scale(ref_scale);
    xform.set_rotation(ref_rotation);

    BOOST_CHECK(xform.translation().isApprox(ref_trans));
    BOOST_CHECK(xform.rotation().isApprox(ref_rotation));
    BOOST_CHECK(xform.scale().isApprox(ref_scale));

    {
        Eigen::Vector3f test_vector(0.f, 0.f, 0.f);
        Eigen::Vector3f ref_vector(0.1f, -0.5f, 2.f);
        BOOST_CHECK((xform * test_vector - test_vector).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector).isZero());
    }

    {
        Eigen::Vector3f test_vector(1.f, 1.f, 1.f);
        Eigen::Vector3f ref_vector_no_trans(1.9510677f, -0.7593853, 2.2083176f);
        Eigen::Vector3f ref_vector_with_trans(2.0510677f, -1.2593853f, 4.2083176f);
        BOOST_CHECK((xform * test_vector - ref_vector_no_trans).isZero());
        BOOST_CHECK((xform.homogeneous_mult(test_vector) - ref_vector_with_trans).isZero());
    }
}

BOOST_AUTO_TEST_CASE(TestCombineTranslation)
{
    cpt::Transform xform1;
    xform1.set_translation(Eigen::Vector3f::UnitX());

    cpt::Transform xform2;
    xform2.set_translation(Eigen::Vector3f::UnitY());

    {
        cpt::Transform test = xform1 * xform2;
        BOOST_CHECK(test.translation().isApprox(Eigen::Vector3f(1.f, 1.f, 0.f)));
    }

    {
        cpt::Transform test = xform2 * xform1;
        BOOST_CHECK(test.translation().isApprox(Eigen::Vector3f(1.f, 1.f, 0.f)));
    }
}

BOOST_AUTO_TEST_CASE(TestCombineScale)
{
    cpt::Transform xform1;
    xform1.set_scale(Eigen::Vector3f::Constant(0.5f));

    cpt::Transform xform2;
    xform2.set_scale(Eigen::Vector3f::Constant(0.1f));

    {
        cpt::Transform test = xform1 * xform2;
        BOOST_CHECK(test.scale().isApproxToConstant(0.05f));
    }

    {
        cpt::Transform test = xform2 * xform1;
        BOOST_CHECK(test.scale().isApproxToConstant(0.05f));
    }
}

BOOST_AUTO_TEST_CASE(TestCombineRotation)
{
    Eigen::AngleAxisf base_ref_rotation1(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());
    Eigen::Matrix3f ref_rotation1 = base_ref_rotation1.toRotationMatrix();

    Eigen::AngleAxisf base_ref_rotation2(M_PI / 3.f, Eigen::Vector3f(1.f, 0.f, 0.f).normalized());
    Eigen::Matrix3f ref_rotation2 = base_ref_rotation2.toRotationMatrix();

    cpt::Transform xform1, xform2;
    xform1.set_rotation(ref_rotation1);
    xform2.set_rotation(ref_rotation2);

    {
        cpt::Transform test = xform1 * xform2;
        BOOST_CHECK(test.rotation().isApprox(ref_rotation1 * ref_rotation2));
    }

    {
        cpt::Transform test = xform2 * xform1;
        BOOST_CHECK(test.rotation().isApprox(ref_rotation2 * ref_rotation1));
    }
}

BOOST_AUTO_TEST_CASE(TestFromTransformMatrix)
{
    const Eigen::Vector3f ref_translation(0.1f, 0.2f, 0.3f);
    const Eigen::Vector3f ref_scale(1.5f, -2.f, 0.75f);
    const Eigen::AngleAxisf ref_rotation(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());

    Eigen::Affine3f ref_xform;
    ref_xform.setIdentity();
    ref_xform.prescale(ref_scale);
    ref_xform.prerotate(ref_rotation);
    ref_xform.pretranslate(ref_translation);

    cpt::Transform xform = cpt::Transform::from_transform_matrix(ref_xform.matrix());
    BOOST_CHECK(xform.translation().isApprox(ref_translation));
    // Can't guarantee that we decompose the scale exactly but rot * scale should be the same.
    BOOST_CHECK((xform.rotation() * xform.scale().asDiagonal()).isApprox(ref_xform.matrix().block(0,0,3,3)));
}

BOOST_AUTO_TEST_CASE(TestCombineTransform)
{
    const Eigen::Vector3f ref_translation1(0.1f, 0.2f, 0.3f);
    const Eigen::Vector3f ref_scale1(1.5f, -2.f, 0.75f);
    const Eigen::AngleAxisf ref_rotation1(M_PI / 4.f, Eigen::Vector3f(1.f, 1.f, 1.f).normalized());

    const Eigen::Vector3f ref_translation2(1.1f, -0.25, -0.1f);
    const Eigen::Vector3f ref_scale2(0.2f, 0.1f, 0.83f);
    const Eigen::AngleAxisf ref_rotation2(M_PI / 3.f, Eigen::Vector3f(0.f, 1.f, 0.5f).normalized());

    Eigen::Affine3f ref_xform;
    ref_xform.setIdentity();
    ref_xform.prescale(ref_scale1);
    ref_xform.prerotate(ref_rotation1);
    ref_xform.pretranslate(ref_translation1);
    ref_xform.prescale(ref_scale2);
    ref_xform.prerotate(ref_rotation2);
    ref_xform.pretranslate(ref_translation2);

    cpt::Transform xform1;
    xform1.set_translation(ref_translation1);
    xform1.set_rotation(ref_rotation1.toRotationMatrix());
    xform1.set_scale(ref_scale1);

    cpt::Transform xform2;
    xform2.set_translation(ref_translation2);
    xform2.set_rotation(ref_rotation2.toRotationMatrix());
    xform2.set_scale(ref_scale2);

    cpt::Transform test_xform = xform2 * xform1;
    BOOST_CHECK(test_xform.translation().isApprox(ref_xform.translation()));
    BOOST_CHECK((test_xform.rotation() * test_xform.scale().asDiagonal()).isApprox(ref_xform.matrix().block(0,0,3,3)));
}
