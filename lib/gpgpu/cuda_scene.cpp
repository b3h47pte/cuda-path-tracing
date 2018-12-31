#include "cuda_scene.h"
#include "gpgpu/cuda_bvh.h"
#include "gpgpu/cuda_converter.h"
#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_utils.h"
#include "utilities/error.h"

namespace cpt {
namespace {

std::vector<CudaGeometry*> extract_cuda_geometry_from_scene(const ScenePtr& scene, bool unpack) {
    std::vector<CudaGeometry*> geom(scene->num_geometry());

    CudaConverter converter;
    for (size_t i = 0; i < geom.size(); ++i) {
        scene->geometry(i)->convert(converter);
        geom[i] = converter.get_from_cache<CudaGeometry>(scene->geometry(i).get());
        CHECK_AND_THROW_ERROR(geom[i] != nullptr, "Failed to convert geometry.");
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

CudaScene::CudaScene(const ScenePtr& scene) {
    _accel_structure = cuda_new<CudaBVH>(extract_cuda_geometry_from_scene(scene, true));
}

CudaScene::~CudaScene() {
    cuda_delete(_accel_structure);
}

}
