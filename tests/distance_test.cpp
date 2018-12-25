#include <math/distance.h>
#include "test_common.h"

namespace {

constexpr double epsilon = 1e-4;

}

TEST(Distance,TestSizeOf)
{
    EXPECT_EQ(sizeof(cpt::Distance), sizeof(float));
}

TEST(Distance,TestConstructorFromMeters)
{
    {
        cpt::Distance dist = cpt::Distance::from_m(0.f);
        EXPECT_NEAR(dist.meters(), 0.f, epsilon);
        EXPECT_NEAR(dist.millimeters(), 0.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_m(-5.f);
        EXPECT_NEAR(dist.meters(), -5.f, epsilon);
        EXPECT_NEAR(dist.millimeters(), -5000.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_m(0.2f);
        EXPECT_NEAR(dist.meters(), 0.2f, epsilon);
        EXPECT_NEAR(dist.millimeters(), 200.f, epsilon);
    }
}

TEST(Distance,TestConstructorFromMillimeters)
{
    {
        cpt::Distance dist = cpt::Distance::from_mm(0.f);
        EXPECT_NEAR(dist.meters(), 0.f, epsilon);
        EXPECT_NEAR(dist.millimeters(), 0.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_mm(-5.f);
        EXPECT_NEAR(dist.meters(), -0.005f, epsilon);
        EXPECT_NEAR(dist.millimeters(), -5.f, epsilon);
    }

    {
        cpt::Distance dist = cpt::Distance::from_mm(0.2f);
        EXPECT_NEAR(dist.meters(), 0.0002f, epsilon);
        EXPECT_NEAR(dist.millimeters(), 0.2f, epsilon);
    }

}

CREATE_GENERIC_TEST_MAIN
