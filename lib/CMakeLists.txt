FIND_PACKAGE(CUDA REQUIRED)

SET(CPT_H_FILES gpgpu/cuda_utils.h
                scene/scene.h
                scene/loader/scene_loader.h)
SET(CPT_CPP_FILES gpgpu/cuda_utils.cpp
                  scene/scene.cpp
                  scene/loader/scene_loader.cpp)

ADD_LIBRARY(cpt STATIC ${CPT_H_FILES} ${CPT_CPP_FILES})
TARGET_INCLUDE_DIRECTORIES(cpt PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(cpt ${CUDA_LIBRARIES})

SET_TARGET_PROPERTIES(cpt PROPERTIES POSITION_INDEPENDENT_CODE ON)
