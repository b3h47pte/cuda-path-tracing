#include "scene.h"

namespace cpt {

Scene::Scene(
    std::vector<GeometryPtr>&& geometry,
    std::unordered_map<std::string, CameraPtr>&& cameras):
    _geometry(std::move(geometry)),
    _cameras(std::move(cameras)) {
}

}
