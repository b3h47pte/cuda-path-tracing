#pragma once

#include <Eigen/Core>

namespace cpt {

Eigen::Matrix3f get_angle_axis_rotation_matrix(const Eigen::Vector3f& angleAxis);
Eigen::Matrix3f get_euler_xyz_rotation_matrix(const Eigen::Vector3f& xyz);

}
