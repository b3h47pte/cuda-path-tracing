#include <math/rotation.h>

#include <Eigen/Geometry>
#include <Eigen/../unsupported/Eigen/EulerAngles>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE RotationTest
#include <boost/test/unit_test.hpp>

namespace {

constexpr double epsilon = 1e-4;

std::vector<Eigen::Vector3f> test_rotation_vectors = {
    {3.f, 1.f, 2.f},
    {0.5f, 0.1f, 0.3f},
    {-0.03f, 2.f, -0.2f}
};

std::vector<Eigen::Vector3f> test_scales = {
    {1.f, 2.f, 0.3f},
    {-1.f, 0.5f, 0.1f},
    {-2.f, -0.1f, -0.8f}
};

}

BOOST_AUTO_TEST_CASE(TestAxisAngleRotation)
{
    for (const auto& vec : test_rotation_vectors) {
        Eigen::AngleAxisf ref_rot(vec.norm(), vec.normalized());
        BOOST_CHECK(cpt::get_angle_axis_rotation_matrix(vec).isApprox(ref_rot.toRotationMatrix()));
    }
}

BOOST_AUTO_TEST_CASE(TestEulerXYZRotation)
{
    for (const auto& vec : test_rotation_vectors) {
        Eigen::EulerAnglesZYXf ref_rot(vec(2), vec(1), vec(0));
        BOOST_CHECK(cpt::get_euler_xyz_rotation_matrix(vec).isApprox(ref_rot.toRotationMatrix()));
    }
}

BOOST_AUTO_TEST_CASE(TestDecomposeScaleRotation)
{
    for (const auto& rot: test_rotation_vectors) {
        Eigen::AngleAxisf angleAxis(rot.norm(), rot.normalized());
        for (const auto& scale : test_scales) {
            Eigen::Affine3f xform;
            xform.setIdentity();
            xform.prescale(scale);
            xform.prerotate(angleAxis);

            Eigen::Matrix3f test_rot;
            Eigen::Vector3f test_scale;
            cpt::decompose_scale_rotation(xform.matrix().block(0,0,3,3), test_rot, test_scale);
            BOOST_CHECK((test_rot * test_scale.asDiagonal()).isApprox(xform.matrix().block(0,0,3,3)));
            BOOST_CHECK_CLOSE(test_rot.determinant(), 1.f, epsilon);
        }
    }
}
