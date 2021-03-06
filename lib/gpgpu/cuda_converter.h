#pragma once

#include "gpgpu/gpgpu_converter.h"

namespace cpt {

class CudaConverter: public GpgpuConverter
{
public:
    void convert(const Triangle& triangle) override;
    void convert(const GeometryAggregate& aggregate) override;
    void convert(const PinholePerspectiveCamera& camera) override;
    void convert(const PointLight& light) override;
};


}
