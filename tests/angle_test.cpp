#include <math/angles.h>

#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE AngleTest
#include <boost/test/unit_test.hpp>

namespace {

constexpr double epsilon = 1e-4;

}

BOOST_AUTO_TEST_CASE(TestConstructorFromDegrees)
{
    {
        cpt::Angle ang = cpt::Angle::from_degrees(0.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 0.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), 0.0f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(45.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 45.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI / 4.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(90.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 90.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI / 2.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(135.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 135.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), 3.0 * M_PI / 4.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(180.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 180.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI, epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestConstructorFromRadians)
{
    {
        cpt::Angle ang = cpt::Angle::from_radians(0.0f);
        BOOST_CHECK_CLOSE(ang.degrees(), 0.0f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), 0.0f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI / 4.f);
        BOOST_CHECK_CLOSE(ang.degrees(), 45.f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI / 4.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI / 2.f);
        BOOST_CHECK_CLOSE(ang.degrees(), 90.f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI / 2.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(3.f * M_PI / 4.f);
        BOOST_CHECK_CLOSE(ang.degrees(), 135.f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), 3.f * M_PI / 4.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI);
        BOOST_CHECK_CLOSE(ang.degrees(), 180.f, epsilon);
        BOOST_CHECK_CLOSE(ang.radians(), M_PI, epsilon);
    }
}
