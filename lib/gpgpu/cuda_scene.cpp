#include "cuda_scene.h"
#include "gpgpu/cuda_bvh.h"
#include "gpgpu/cuda_converter.h"
#include "gpgpu/cuda_camera.h"
#include "gpgpu/cuda_geometry.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_utils.h"
#include "utilities/error.h"
#include "utilities/timer.h"

namespace cpt {
namespace {

std::vector<CudaGeometry*> extract_cuda_geometry_from_scene(const ScenePtr& scene, bool unpack) {
    std::vector<CudaGeometry*> geom(scene->num_geometry());

    CudaConverter converter;
    LOG("Total Geometry to Convert: " << geom.size(), LogLevel::Debug);
    START_TIMER_DEBUG(convert, "Converting to CUDA...");
    for (size_t i = 0; i < geom.size(); ++i) {
        scene->geometry(i)->convert(converter);
        geom[i] = converter.get_from_cache<CudaGeometry>(scene->geometry(i).get());
        CHECK_AND_THROW_ERROR(geom[i] != nullptr, "Failed to convert geometry.");
    }
    END_TIMER(convert);

    if (!unpack) {
        return geom;
    }

    CudaGeometryAggregate cuda_agg(geom);

    std::vector<CudaGeometry*> unpacked_geom;
    cuda_agg.unpack_geometry(unpacked_geom);
    return unpacked_geom;
}

}

CudaScene::CudaScene(const ScenePtr& scene, const std::string& camera_id) {
    _accel_structure = cuda_new<CudaBVH>(extract_cuda_geometry_from_scene(scene, true));

    CHECK_AND_THROW_ERROR(scene->has_camera(camera_id), "Invalid camera ID for render.");
    CudaConverter converter;

    const auto& camera = scene->camera(camera_id);
    camera->convert(converter);
    _render_camera = converter.get_from_cache<CudaCamera>(camera.get());

    _num_lights = scene->num_lights();
    _lights = cuda_new_array<CudaLight*>(scene->num_lights());   
    for (size_t i = 0; i < scene->num_lights(); ++i) {
        scene->light(i)->convert(converter);
        _lights[i] = converter.get_from_cache<CudaLight>(scene->light(i).get());
    }
}

CudaScene::~CudaScene() {
    cuda_delete(_accel_structure);
    cuda_delete(_render_camera);
    cuda_delete_array(_lights, num_lights());
}

void CudaScene::generate_rays(CudaSampler* samplers, CudaRay* rays, size_t width, size_t height) const {
    _render_camera->generate_rays(samplers, rays, width, height);
}

}
