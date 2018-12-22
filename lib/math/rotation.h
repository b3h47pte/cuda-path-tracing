#pragma once

#include <Eigen/Core>

namespace cpt {

Eigen::Matrix3f get_angle_axis_rotation_matrix(const Eigen::Vector3f& angleAxis);

// Euler angles use extrinsic rotations.
Eigen::Matrix3f get_euler_xyz_rotation_matrix(const Eigen::Vector3f& xyz);

void decompose_scale_rotation(const Eigen::Matrix3f& rotScaleMat, Eigen::Matrix3f& rot, Eigen::Vector3f& scale);

}
