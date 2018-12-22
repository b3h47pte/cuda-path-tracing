#include "transform.h"

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
    _scale = newRotScaleMat.colwise().norm();
    _rotation = newRotScaleMat.array().colwise() / _scale.array();
    return *this;
}

Transform Transform::operator*(const Transform& other) const {
    Transform xform = *this;
    xform *= other;
    return xform;
}

}
