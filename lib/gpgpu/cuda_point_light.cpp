#include "cuda_point_light.h"

namespace cpt {

CudaPointLight::CudaPointLight():
    CudaLight(CudaLight::Type::Point) {
}

}
