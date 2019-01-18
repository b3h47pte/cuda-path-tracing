#pragma once

#include "scene/lights/light.h"

namespace cpt {

class PointLight : public Light
{
public:
    void convert(GpgpuConverter& converter) const override;
};

}
