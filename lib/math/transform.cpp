#include "transform.h"

#include <Eigen/Dense>
#include "math/rotation.h"
#include "utilities/error.h"

namespace cpt {

Transform::Transform() {
    reset();
}

void Transform::reset() {
    _scale.setOnes();
    _rotation.setIdentity();
    _translation.setZero();
}

void Transform::set_translation(const Eigen::Vector3f& trans) {
    _translation = trans;
}

void Transform::set_rotation(const Eigen::Matrix3f& rot) {
    // Check to make sure we have a valid rotation.
    CHECK_AND_THROW_ERROR(std::abs(rot.determinant() - 1.f) < 1e-6, "Invalid rotation matrix [Determinant:" <<  rot.determinant() << "].");
    _rotation = rot;
}

void Transform::set_scale(const Eigen::Vector3f& scale) {
    _scale = scale;
}

Eigen::Vector3f Transform::operator*(const Eigen::Vector3f& other) const {
    return (_rotation * _scale.cwiseProduct(other));
}

Eigen::Vector3f Transform::homogeneous_mult(const Eigen::Vector3f& other) const {
    return (*this * other + _translation);
}

Transform& Transform::operator*=(const Transform& other) {
    // Assume that we have
    // T_1 R_1 S_1 for one transform and T_2 R_2 S_2 for the other transform.
    // The new transform would thus have a translation of T_1 + R_1 S_1 T_2
    // Its rotation/scale would be a combined R_1 S_1 R_2 S_2. We can extract
    // the new scale by taking the magnitude of the colums and the rotation
    // matrix are just the normalized columns.
    _translation += (*this * other._translation);

    Eigen::Matrix3f newRotScaleMat = _rotation * _scale.asDiagonal() * other._rotation * other._scale.asDiagonal();
    decompose_scale_rotation(newRotScaleMat, _rotation, _scale);
    return *this;
}

Transform Transform::operator*(const Transform& other) const {
    Transform xform = *this;
    xform *= other;
    return xform;
}

Transform Transform::from_transform_matrix(const Eigen::Matrix4f& matrix) {
    Transform xform;
    xform.set_translation(matrix.block(0, 3, 3, 1));

    Eigen::Matrix3f rot;
    Eigen::Vector3f scale;
    decompose_scale_rotation(matrix.block(0,0,3,3), rot, scale);    

    xform.set_scale(scale);
    xform.set_rotation(rot);
    return xform;
}

Transform Transform::inverse() const {
    Eigen::Matrix4f mat = to_matrix();
    Eigen::Matrix4f new_mat = mat;
    new_mat.block(0,0,3,3) = mat.block(0,0,3,3).inverse();
    new_mat.block(0,3,3,1) = -new_mat.block(0,0,3,3) * mat.block(0,3,3, 1);
    return Transform::from_transform_matrix(new_mat);
}

Eigen::Matrix4f Transform::to_matrix() const {
    Eigen::Matrix4f matrix;
    matrix.setIdentity();
    matrix.block(0,0,3,3) = _rotation * _scale.asDiagonal();
    matrix.block(0,3,3,1) = _translation;
    return matrix;
}

}
