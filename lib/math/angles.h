#pragma once

namespace cpt {

// Store in radians and convert as necessary.
class Angle
{
public:
    static Angle from_radians(float rad);
    static Angle from_degrees(float deg);

private:
    Angle(float radians):
        _radians(radians)
    {}

    float _radians;
};

}
