#include "scene.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "gpgpu/cuda_ptr.h"
#include <unordered_set>

namespace cpt {

Scene::Scene(
    std::vector<GeometryPtr>&& geometry,
    std::unordered_map<std::string, CameraPtr>&& cameras):
    _geometry(std::move(geometry)),
    _cameras(std::move(cameras)) {
}

std::vector<CudaGeometry*> Scene::cuda_geometry(bool unpack) const {
    std::vector<CudaGeometry*> geom(_geometry.size());

    // NOTE: This cache is currently not thread safe.
    Geometry::CudaGeometryCache cache;
    for (size_t i = 0; i < geom.size(); ++i) {
        geom[i] = _geometry[i]->create_cuda(cache);
    }

    if (!unpack) {
        return geom;
    }

    CudaGeometryAggregate cuda_agg(geom);

    std::vector<CudaGeometry*> unpacked_geom;
    cuda_agg.unpack_geometry(unpacked_geom);
    return unpacked_geom;
}

}
