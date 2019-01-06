#include "cuda_object.h"

#include "gpgpu/math/cuda_matrix.h"

namespace cpt {

void CudaObject::bake_from_object(const Object& object) {
    Eigen::Vector3f cpuPosition = object.object_to_world_xform().homogeneous_mult(Eigen::Vector3f::Zero());
    _position = eigen_vector_to_cuda<float,3>(cpuPosition);
    _object_to_world_xform = CudaAffineTransform(object.object_to_world_xform());
    _world_to_object_xform = CudaAffineTransform(object.object_to_world_xform().inverse());

    _world_up_dir = eigen_vector_to_cuda<float,3>(object.world_up_dir());
    _world_right_dir = eigen_vector_to_cuda<float,3>(object.world_right_dir());
    _world_forward_dir = eigen_vector_to_cuda<float,3>(object.world_forward_dir());
}

}
