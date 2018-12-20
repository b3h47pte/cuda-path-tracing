#pragma once

#include <Eigen/Core>

namespace cpt {

class Transform
{
public:
    Transform();

    // Set to be equivalent to the identity matrix.
    void reset();

    Eigen::Vector3d operator*(const Eigen::Vector3d& other) const;
    Eigen::Vector3d homogeneousMult(const Eigen::Vector3d& other) const;

    Transform& operator*=(const Transform& other);
    Transform operator*(const Transform& other) const;

private:
    // Scale, rotation, translation.
    // Keep rotation in matrix mode since I don't particularly care for supporting
    // more fanciful operations that'd require storing quaternions.
    Eigen::Vector3d _scale;
    Eigen::Matrix3d _rotation;
    Eigen::Vector3d _translation;
};


}
