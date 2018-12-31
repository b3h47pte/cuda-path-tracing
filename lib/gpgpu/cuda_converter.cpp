#include "cuda_converter.h"

#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_geometry_aggregate.h"
#include "gpgpu/cuda_triangle.h"
#include "gpgpu/math/cuda_vector.h"

#include "scene/geometry/triangle.h"
#include "scene/geometry/geometry_aggregate.h"
#include <tbb/tbb.h>
#include "utilities/log.h"
#include "utilities/timer.h"
#include "utilities/progress_bar.h"

#include <vector>

namespace cpt {

void CudaConverter::convert(const Triangle& triangle) {
    // TODO: This function is slower than it has to be. I'm going to guess
    // that it has to do with the excessivea mount of memory operations being
    // done due to the use of CudaVector.
    auto* key = const_cast<Triangle*>(&triangle);
    auto* ptr = get_from_cache<CudaGeometry>(key);
    if (ptr) {
        return;
    }

    std::vector<CudaVector<float,3>*> positions(3);
    std::vector<CudaVector<float,3>*> normals(3);
    std::vector<CudaVector<float,2>*> uvs(3);

    for (auto i = 0; i < 3; ++i) {
        positions[i] = eigen_vector_to_cuda<float,3>(triangle.get_vertex_container()->position(triangle.get_vertex_indices()(i)));
        normals[i] = eigen_vector_to_cuda<float,3>(triangle.get_vertex_container()->normal(triangle.get_normal_indices()(i)));
        uvs[i] = eigen_vector_to_cuda<float,2>(triangle.get_vertex_container()->uv(triangle.get_uv_indices()(i)));
    }

    ptr = cuda_new<CudaTriangle>(positions, normals, uvs);
    add_to_cache(key, ptr);
}

void CudaConverter::convert(const GeometryAggregate& aggregate) {
    auto* key = const_cast<GeometryAggregate*>(&aggregate);
    auto* ptr = get_from_cache<CudaGeometry>(key);
    if (ptr) {
        return;
    }

    LOG("Total Children to Convert: " << aggregate.num_children(), LogLevel::Debug);
    START_TIMER_DEBUG(convert, "Converting children to CUDA...");
    std::vector<CudaGeometry*> cuda_children(aggregate.num_children());
    ProgressBar progress(cuda_children.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, cuda_children.size()),
    [this, &cuda_children, &aggregate, &progress](const tbb::blocked_range<size_t>& r){
        for (size_t i = r.begin(); i < r.end(); ++i) {
            progress.job_done();
            aggregate.get_geometry(i)->convert(*this);
            cuda_children[i] = get_from_cache<CudaGeometry>(aggregate.get_geometry(i).get());
        }
    });
    END_TIMER(convert);
    progress.complete();

    ptr = cuda_new<CudaGeometryAggregate>(cuda_children);
    add_to_cache(key, ptr);
}

}
