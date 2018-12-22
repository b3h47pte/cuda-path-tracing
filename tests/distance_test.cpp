#include <math/distance.h>
#include "test_common.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DistanceTest
#include <boost/test/unit_test.hpp>

namespace {

constexpr double epsilon = 1e-4;

}

BOOST_AUTO_TEST_CASE(TestSizeOf)
{
    BOOST_CHECK_EQUAL(sizeof(cpt::Distance), sizeof(float));
}

BOOST_AUTO_TEST_CASE(TestConstructorFromMeters)
{
    {
        cpt::Distance dist = cpt::Distance::from_m(0.f);
        BOOST_CHECK_CLOSE(dist.meters(), 0.f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), 0.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_m(-5.f);
        BOOST_CHECK_CLOSE(dist.meters(), -5.f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), -5000.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_m(0.2f);
        BOOST_CHECK_CLOSE(dist.meters(), 0.2f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), 200.f, epsilon);
    }
}

BOOST_AUTO_TEST_CASE(TestConstructorFromMillimeters)
{
    {
        cpt::Distance dist = cpt::Distance::from_mm(0.f);
        BOOST_CHECK_CLOSE(dist.meters(), 0.f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), 0.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_mm(-5.f);
        BOOST_CHECK_CLOSE(dist.meters(), -0.005f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), -5.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_mm(0.2f);
        BOOST_CHECK_CLOSE(dist.meters(), 0.0002f, epsilon);
        BOOST_CHECK_CLOSE(dist.millimeters(), 0.2f, epsilon);
    }

}
