#include "scene.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "gpgpu/cuda_ptr.h"
#include <unordered_set>

namespace cpt {

Scene::Scene(
    std::vector<GeometryPtr>&& geometry,
    std::unordered_map<std::string, CameraPtr>&& cameras,
    std::vector<LightPtr>&& lights):
    _geometry(std::move(geometry)),
    _cameras(std::move(cameras)),
    _lights(std::move(lights)) {
}

}
