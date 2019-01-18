#include "point_light.h"

#include "gpgpu/gpgpu_converter.h"

namespace cpt {

void PointLight::convert(GpgpuConverter& converter) const {
    converter.convert(*this);
}

}
