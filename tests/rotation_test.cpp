#include <math/rotation.h>

#include <Eigen/Geometry>
#include <Eigen/../unsupported/Eigen/EulerAngles>
#include "test_common.h"

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

TEST(Rotation,TestAxisAngleRotation)
{
    for (const auto& vec : test_rotation_vectors) {
        Eigen::AngleAxisf ref_rot(vec.norm(), vec.normalized());
        EXPECT_TRUE(cpt::get_angle_axis_rotation_matrix(vec).isApprox(ref_rot.toRotationMatrix()));
    }
}

TEST(Rotation,TestEulerXYZRotation)
{
    for (const auto& vec : test_rotation_vectors) {
        Eigen::EulerAnglesZYXf ref_rot(vec(2), vec(1), vec(0));
        EXPECT_TRUE(cpt::get_euler_xyz_rotation_matrix(vec).isApprox(ref_rot.toRotationMatrix()));
    }
}

TEST(Rotation,TestDecomposeScaleRotation)
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
            EXPECT_TRUE((test_rot * test_scale.asDiagonal()).isApprox(xform.matrix().block(0,0,3,3)));
            EXPECT_NEAR(test_rot.determinant(), 1.f, epsilon);
        }
    }
}

CREATE_GENERIC_TEST_MAIN
