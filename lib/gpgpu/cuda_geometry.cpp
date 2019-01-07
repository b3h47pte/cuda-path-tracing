#include "cuda_geometry.h"

#include <atomic>

namespace cpt {
namespace {

std::atomic_size_t geometry_id_counter(0);

}

CudaGeometry::CudaGeometry(Type type):
    _type(type),
    _id(geometry_id_counter++) {
}

void CudaGeometry::bake_from_object(const Object& object) {
    CudaObject::bake_from_object(object);
    set_world_space_aabb(_aabb.transform(object_to_world_xform()));
}

}
