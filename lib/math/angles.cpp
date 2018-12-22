#include "angles.h"
#include <cmath>

namespace cpt {

Angle Angle::from_radians(float rad)
{
    return Angle(rad);
}

Angle Angle::from_degrees(float deg)
{
    return Angle(deg * M_PI / 180.0f);
}

}
