#include "rotation.h"

#include <Eigen/Geometry>

namespace cpt {

Eigen::Matrix3f get_angle_axis_rotation_matrix(const Eigen::Vector3f& angleAxis) {
    return Eigen::AngleAxisf(angleAxis.norm(), angleAxis.normalized()).toRotationMatrix();
}

Eigen::Matrix3f get_euler_xyz_rotation_matrix(const Eigen::Vector3f& xyz) {
    return get_angle_axis_rotation_matrix(xyz(2) * Eigen::Vector3f::UnitZ()) *
        get_angle_axis_rotation_matrix(xyz(1) * Eigen::Vector3f::UnitY()) *
        get_angle_axis_rotation_matrix(xyz(0) * Eigen::Vector3f::UnitX());
}

}
