#include <scene/camera/pinhole_perspective_camera.h>
#include "test_common.h"

namespace {

constexpr double epsilon = 1e-4;

}

TEST(PinholePerspectiveCamera,TestConstructor)
{
    cpt::PinholePerspectiveCamera camera(
        cpt::Angle::from_degrees(45.0f),  // fov
        1.5f,                             // aspect ratio
        cpt::Distance::from_mm(35.f),     // focal length
        cpt::Distance::from_m(0.01f),     // near z 
        cpt::Distance::from_m(1000.f));   // far z

    EXPECT_NEAR(camera.horizontal_fov().degrees(), 45.f, epsilon);
    EXPECT_NEAR(camera.film_aspect_ratio(), 1.5f, epsilon);
    EXPECT_NEAR(camera.focal_length().millimeters(), 35.f, epsilon);
    EXPECT_NEAR(camera.near_z().meters(), 0.01f, epsilon);
    EXPECT_NEAR(camera.far_z().meters(), 1000.f, epsilon);
}

CREATE_GENERIC_TEST_MAIN
