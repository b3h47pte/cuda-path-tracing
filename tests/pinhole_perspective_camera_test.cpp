#include <scene/camera/pinhole_perspective_camera.h>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PinholePerspectiveCameraTest
#include <boost/test/unit_test.hpp>

namespace {

constexpr double epsilon = 1e-4;

}

BOOST_AUTO_TEST_CASE(TestConstructor)
{
    cpt::PinholePerspectiveCamera camera(
        cpt::Angle::from_degrees(45.0f),  // fov
        1.5f,                             // aspect ratio
        cpt::Distance::from_mm(35.f),     // focal length
        cpt::Distance::from_m(0.01f),     // near z 
        cpt::Distance::from_m(1000.f));   // far z

    BOOST_CHECK_CLOSE(camera.horizontal_fov().degrees(), 45.f, epsilon);
    BOOST_CHECK_CLOSE(camera.film_aspect_ratio(), 1.5f, epsilon);
    BOOST_CHECK_CLOSE(camera.focal_length().millimeters(), 35.f, epsilon);
    BOOST_CHECK_CLOSE(camera.near_z().meters(), 0.01f, epsilon);
    BOOST_CHECK_CLOSE(camera.far_z().meters(), 1000.f, epsilon);
}
