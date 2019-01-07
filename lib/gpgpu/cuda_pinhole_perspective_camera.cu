#include "cuda_pinhole_perspective_camera.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {
namespace {

__global__ void generate_pinhole_perspective_rays(CudaSampler* samplers, const CudaPinholePerspectiveCamera* camera, CudaRay* rays, size_t width, size_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int flat_idx = y * width + x;

    if (flat_idx >= width * height) {
        return;
    }

    CudaRay& active_ray = rays[flat_idx];
    active_ray.dbg_set_x_idx(x);
    active_ray.dbg_set_y_idx(y);
    active_ray.dbg_set_flat_idx(flat_idx);

    CudaSampler& sampler = samplers[flat_idx];

    // Offset from pixel center.
    CudaVector<float,2> pixel_sample;
    sampler.sample_2d(
        pixel_sample,
        CudaVector<float,2>::Constant(-.5f),
        CudaVector<float,2>::Constant(.5f));

    const CudaVector<float,3>& camera_forward = camera->world_forward_dir();
    const CudaVector<float,3>& camera_right = camera->world_right_dir();
    const CudaVector<float,3>& camera_up = camera->world_up_dir();

    // Compute ray origin.
    // Note that (0, 0) is the top left corner. Since we're putting the film plane
    // "behind" the aperture, this means that top left corner of the image is the
    // bottom right corner of the film plane.
    CudaVector<float,3> ray_origin = camera->position();
    ray_origin -= camera_forward * camera->focal_length_m();

    const float x_offset_px = static_cast<float>(x) + 0.5f + pixel_sample[0] - width / 2.f;
    const float y_offset_px = static_cast<float>(y) + 0.5f + pixel_sample[1] - height / 2.f;

    const float film_width = 
        2.f * camera->focal_length_m() * ::tan(camera->horizontal_fov_radians() / 2.f);
    const float film_height = film_width / camera->aspect_ratio();

    const float x_offset = x_offset_px / width * film_width;
    const float y_offset = y_offset_px / height * film_height;

    ray_origin -= camera_right * x_offset;
    ray_origin -= camera_up * y_offset;

    // Make the ray pass through the camera location.
    const CudaVector<float,3>& ray_target = camera->position();
    active_ray.set_direction(ray_target - ray_origin);

    // Move origin to ray_target so we don't hit anything before we exit the pinhole.
    // Then offset by near_z so we don't hit anything before the near_z plane.
    ray_origin = ray_target;
    ray_origin += active_ray.direction() * camera->near_z_m();

    active_ray.set_origin(ray_origin);
    active_ray.set_max_t(camera->far_z_m() - camera->near_z_m());
}

}

void CudaPinholePerspectiveCamera::generate_rays(CudaSampler* samplers, CudaRay* rays, size_t width, size_t height) const {
    dim3 blocks, threads;
    compute_blocks_threads_2d(blocks, threads, width, height);
    generate_pinhole_perspective_rays<<<blocks, threads>>>(samplers, this, rays, width, height);
}

}
