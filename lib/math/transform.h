#pragma once

#include <Eigen/Core>

namespace cpt {

class Transform
{
public:
    Transform();

    // Set to be equivalent to the identity matrix.
    void reset();

    void set_translation(const Eigen::Vector3f& trans);
    const Eigen::Vector3f& translation() const { return _translation; }

    void set_rotation(const Eigen::Matrix3f& rot);
    const Eigen::Matrix3f& rotation() const { return _rotation; }

    void set_scale(const Eigen::Vector3f& scale);
    const Eigen::Vector3f& scale() const { return _scale; }

    Eigen::Vector3f operator*(const Eigen::Vector3f& other) const;
    Eigen::Vector3f homogeneous_mult(const Eigen::Vector3f& other) const;

    Transform& operator*=(const Transform& other);
    Transform operator*(const Transform& other) const;

private:
    // Scale, rotation, translation.
    // Keep rotation in matrix mode since I don't particularly care for supporting
    // more fanciful operations that'd require storing quaternions.
    Eigen::Vector3f _scale;
    Eigen::Matrix3f _rotation;
    Eigen::Vector3f _translation;
};


}
