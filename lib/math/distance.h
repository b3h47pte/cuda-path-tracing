#pragma once

namespace cpt {

// Store in meters and convert as necessary.
class Distance
{
public:
    static Distance from_mm(float mm);
    static Distance from_m(float m);

    float meters() const { return _meters; }
    float millimeters() const { return _meters * 1000.f; }

private:
    Distance(float meters):
        _meters(meters)
    {}
    float _meters;
};

}
