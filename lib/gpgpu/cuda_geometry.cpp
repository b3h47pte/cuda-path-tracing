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

}
