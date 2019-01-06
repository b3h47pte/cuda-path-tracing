#pragma once

#include <Eigen/Core>

namespace cpt {

class Transform
{
public:
    Transform();

    static Transform from_transform_matrix(const Eigen::Matrix4f& matrix);

    Transform(const Transform&) = default;
    Transform(Transform&&) = default;
    Transform& operator=(const Transform&) = default;
    Transform& operator=(Transform&&) = default;
    ~Transform() = default;

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

    Transform inverse() const;
    Eigen::Matrix4f to_matrix() const;

private:
    // Scale, rotation, translation.
    // Keep rotation in matrix mode since I don't particularly care for supporting
    // more fanciful operations that'd require storing quaternions.
    Eigen::Vector3f _scale;
    Eigen::Matrix3f _rotation;
    Eigen::Vector3f _translation;
};

}
