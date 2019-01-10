#pragma once

#include <Eigen/Core>
#include <memory>
#include "scene/object.h"

namespace cpt {

class Light: public Object
{
public:
    Light();

    void set_color(const Eigen::Vector3f& c) { _color = c; }
    const Eigen::Vector3f& color() const { return _color; }

private:
    Eigen::Vector3f _color;
};

using LightPtr = std::shared_ptr<Light>;

}
