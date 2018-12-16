FIND_PACKAGE(CUDA REQUIRED)

SET(CPT_H_FILES gpgpu/cuda_utils.h
                scene/scene.h
                scene/loader/scene_loader.h
                scene/loader/mesh_loader.h
                scene/geometry/geometry.h
                scene/geometry/geometry_aggregate.h
                scene/geometry/vertex_container.h
                scene/geometry/triangle.h
                scene/object.h
                utilities/error.h)
SET(CPT_CPP_FILES gpgpu/cuda_utils.cpp
                  scene/scene.cpp
                  scene/geometry/geometry.cpp
                  scene/geometry/geometry_aggregate.cpp
                  scene/geometry/vertex_container.cpp
                  scene/geometry/triangle.cpp
                  scene/object.cpp
                  scene/loader/scene_loader.cpp
                  scene/loader/mesh_loader.cpp)

ADD_LIBRARY(cpt STATIC ${CPT_H_FILES} ${CPT_CPP_FILES})

TARGET_INCLUDE_DIRECTORIES(cpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} )
TARGET_LINK_LIBRARIES(cpt ${CUDA_LIBRARIES})

# Boost
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(cpt Boost::system Boost::filesystem)

# Eigen
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
MESSAGE(STATUS ${EIGEN3_INCLUDE_DIR})

# External
TARGET_INCLUDE_DIRECTORIES(cpt SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/external/include)

SET_TARGET_PROPERTIES(cpt PROPERTIES POSITION_INDEPENDENT_CODE ON)
