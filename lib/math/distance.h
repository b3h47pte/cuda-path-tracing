#pragma once

namespace cpt {

// Store in meters and convert as necessary.
class Distance
{
public:
    static Distance from_mm(float mm);
    static Distance from_m(float m);

private:
    Distance(float meters):
        _meters(meters)
    {}
    float _meters;
};

}
