#pragma once

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <math/transform.h>
#include <scene/camera/pinhole_perspective_camera.h>
#include <scene/geometry/geometry.h>
#include <iostream>

// TODO: Rename..
extern Eigen::Array3Xf ref_positions;
extern Eigen::Array2Xf ref_uvs;
extern Eigen::Array3Xf ref_normals;

void check_test_plane(const cpt::GeometryPtr& geom);
void check_pinhole_perspective_camera_equal(
    const cpt::PinholePerspectiveCameraPtr& cam1,
    const cpt::PinholePerspectiveCameraPtr& cam2,
    double epsilon);
void check_transforms_equal(
    const cpt::Transform& xform1,
    const cpt::Transform& xform2,
    double epsilon);
cpt::GeometryPtr construct_test_plane();

class TestCamera: public cpt::Camera
{};

#define CREATE_GENERIC_TEST_MAIN \
int main(int argc, char** argv) {\
    ::testing::InitGoogleTest(&argc, argv);\
    ::testing::InitGoogleMock(&argc, argv);\
    return RUN_ALL_TESTS();\
}
