FIND_PACKAGE(CUDA REQUIRED)

SET(CPT_H_FILES gpgpu/cuda_aabb.h
                gpgpu/cuda_acceleration_structure.h
                gpgpu/cuda_bvh.h
                gpgpu/cuda_converter.h
                gpgpu/cuda_utils.h
                gpgpu/cuda_scene.h
                gpgpu/cuda_triangle.h
                gpgpu/cuda_geometry.h
                gpgpu/cuda_geometry_aggregate.h
                gpgpu/math/cuda_vector.h
                gpgpu/gpgpu_converter.h
                math/angles.h
                math/distance.h
                math/rotation.h
                math/transform.h
                render/aov_output.h
                render/cuda_renderer.h
                gpgpu/cuda_ptr.h
                scene/scene.h
                scene/camera/camera.h
                scene/camera/pinhole_perspective_camera.h
                scene/loader/scene_loader.h
                scene/loader/mesh_loader.h
                scene/loader/camera_loader.h
                scene/loader/xform_loader.h
                scene/geometry/geometry.h
                scene/geometry/vertex_geometry.h
                scene/geometry/geometry_aggregate.h
                scene/geometry/vertex_container.h
                scene/geometry/triangle.h
                scene/object.h
                utilities/eigen_utility.h
                utilities/error.h
                utilities/filesystem_utility.h
                utilities/json_utility.h
                utilities/noncopyable.h
                utilities/memory_ownership.h)

SET(CPT_CPP_FILES gpgpu/cuda_aabb.cpp
                  gpgpu/cuda_acceleration_structure.cpp
                  gpgpu/cuda_bvh.cpp
                  gpgpu/cuda_converter.cpp
                  gpgpu/cuda_utils.cpp
                  gpgpu/cuda_scene.cpp
                  gpgpu/cuda_triangle.cpp
                  gpgpu/cuda_geometry.cpp
                  gpgpu/cuda_geometry_aggregate.cpp
                  math/angles.cpp
                  math/distance.cpp
                  math/rotation.cpp
                  math/transform.cpp
                  render/aov_output.cpp
                  render/cuda_renderer.cpp
                  scene/scene.cpp
                  scene/camera/camera.cpp
                  scene/camera/pinhole_perspective_camera.cpp
                  scene/geometry/geometry.cpp
                  scene/geometry/geometry_aggregate.cpp
                  scene/geometry/vertex_container.cpp
                  scene/geometry/triangle.cpp
                  scene/object.cpp
                  scene/loader/scene_loader.cpp
                  scene/loader/mesh_loader.cpp
                  scene/loader/camera_loader.cpp
                  scene/loader/xform_loader.cpp
                  utilities/filesystem_utility.cpp
                  utilities/json_utility.cpp)

SET(CPT_CU_FILES render/cuda_renderer.cu)

ADD_LIBRARY(cpt SHARED ${CPT_H_FILES} ${CPT_CPP_FILES} ${CPT_CU_FILES})

TARGET_INCLUDE_DIRECTORIES(cpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} )
TARGET_LINK_LIBRARIES(cpt ${CUDA_LIBRARIES})
CUDA_ADD_CUBLAS_TO_TARGET(cpt)

# Boost
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(cpt Boost::system Boost::filesystem)

# Eigen
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})

# External
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/external/include)

# Image IO
FIND_PACKAGE(TIFF REQUIRED)
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${TIFF_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(cpt ${TIFF_LIBRARIES})

SET_TARGET_PROPERTIES(cpt PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(cpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
