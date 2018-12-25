#include <math/angles.h>

#include <cmath>
#include "test_common.h"

namespace {

constexpr double epsilon = 1e-4;

}

TEST(Angle,TestSizeOf)
{
    EXPECT_EQ(sizeof(cpt::Angle), sizeof(float));
}

TEST(Angle,TestConstructorFromDegrees)
{
    {
        cpt::Angle ang = cpt::Angle::from_degrees(0.0f);
        EXPECT_NEAR(ang.degrees(), 0.0f, epsilon);
        EXPECT_NEAR(ang.radians(), 0.0f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(45.0f);
        EXPECT_NEAR(ang.degrees(), 45.0f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI / 4.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(90.0f);
        EXPECT_NEAR(ang.degrees(), 90.0f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI / 2.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(135.0f);
        EXPECT_NEAR(ang.degrees(), 135.0f, epsilon);
        EXPECT_NEAR(ang.radians(), 3.0 * M_PI / 4.0, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_degrees(180.0f);
        EXPECT_NEAR(ang.degrees(), 180.0f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI, epsilon);
    }
}

TEST(Angle,TestConstructorFromRadians)
{
    {
        cpt::Angle ang = cpt::Angle::from_radians(0.0f);
        EXPECT_NEAR(ang.degrees(), 0.0f, epsilon);
        EXPECT_NEAR(ang.radians(), 0.0f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI / 4.f);
        EXPECT_NEAR(ang.degrees(), 45.f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI / 4.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI / 2.f);
        EXPECT_NEAR(ang.degrees(), 90.f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI / 2.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(3.f * M_PI / 4.f);
        EXPECT_NEAR(ang.degrees(), 135.f, epsilon);
        EXPECT_NEAR(ang.radians(), 3.f * M_PI / 4.f, epsilon);
    }

    {
        cpt::Angle ang = cpt::Angle::from_radians(M_PI);
        EXPECT_NEAR(ang.degrees(), 180.f, epsilon);
        EXPECT_NEAR(ang.radians(), M_PI, epsilon);
    }
}

CREATE_GENERIC_TEST_MAIN
