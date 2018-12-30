#include "scene.h"

namespace cpt {

Scene::Scene(
    std::vector<GeometryPtr>&& geometry,
    std::unordered_map<std::string, CameraPtr>&& cameras):
    _geometry(std::move(geometry)),
    _cameras(std::move(cameras)) {
}

std::vector<CudaGeometry*> Scene::cuda_geometry() const {
    std::vector<CudaGeometry*> geom(_geometry.size());

    // NOTE: This cache is currently not thread safe.
    Geometry::CudaGeometryCache cache;
    for (size_t i = 0; i < geom.size(); ++i) {
        geom[i] = _geometry[i]->create_cuda(cache);
    }
    return geom;
}

}
