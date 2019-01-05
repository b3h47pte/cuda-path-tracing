#include "cuda_pinhole_perspective_camera.h"

namespace cpt {

CudaPinholePerspectiveCamera::CudaPinholePerspectiveCamera(const PinholePerspectiveCamera& camera):
    _camera(camera) {

}

}
