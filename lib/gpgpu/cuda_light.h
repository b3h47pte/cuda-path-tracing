#pragma once

#include "gpgpu/cuda_object.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaLight: public CudaObject
{
public:
    enum class Type
    {
        Point
    };

    CudaLight(Type type):
        _type(type) {
    }

    Type type() const { return _type; }

    CUDA_HOST void set_color(const CudaVector<float,3>& color) { _color = color; }
    CUDA_DEVHOST const CudaVector<float,3>& color() const { return _color; }

private:
    Type _type;

    CudaVector<float,3> _color;
};

}
