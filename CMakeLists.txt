CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)
PROJECT("cuda-path-tracing" LANGUAGES CXX CUDA)
ENABLE_TESTING()

IF(NOT CMAKE_BUILD_TYPE) 
    SET(CMAKE_BUILD_TYPE Release) 
ENDIF() 

# Extra CMake stuff.
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Compilation Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNO_CUDA_CHECKS")
SET(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CUDA_STANDARD 11)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

# This is probably excessive.
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Options
OPTION(WITH_PROGRAMS "Compile programs." ON)
OPTION(WITH_TEST_ASSETS "Download and setup test assets." ON)
OPTION(WITH_TESTS "Compile tests." ON)

# Setup test assets.
IF(WITH_TEST_ASSETS)
    # Sponza Atrium
    MESSAGE(STATUS "Downloading Sponza Atrium...")
    SET(SPONZA_FNAME ${CMAKE_CURRENT_SOURCE_DIR}/assets/sponza.zip)
    IF(NOT EXISTS ${SPONZA_FNAME})
        FILE(DOWNLOAD https://casual-effects.com/g3d/data10/common/model/crytek_sponza/sponza.zip ${SPONZA_FNAME} SHOW_PROGRESS)
    ENDIF()

    MESSAGE(STATUS "Downloading San Miguel...")
    SET(SAN_MIGUEL_FNAME ${CMAKE_CURRENT_SOURCE_DIR}/assets/San_Miguel.zip)
    IF(NOT EXISTS ${SAN_MIGUEL_FNAME})
        FILE(DOWNLOAD https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip ${SAN_MIGUEL_FNAME} SHOW_PROGRESS)
    ENDIF()

    MESSAGE(STATUS "Downloading Sibenik Cathedral...")
    SET(SIBENIK_FNAME ${CMAKE_CURRENT_SOURCE_DIR}/assets/sibenik.zip)
    IF(NOT EXISTS ${SIBENIK_FNAME})
        FILE(DOWNLOAD https://casual-effects.com/g3d/data10/research/model/sibenik/sibenik.zip ${SIBENIK_FNAME} SHOW_PROGRESS)
    ENDIF()
ENDIF()

# Packages
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
FIND_PACKAGE(Boost 1.60 REQUIRED
             COMPONENTS system filesystem program_options unit_test_framework log)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(TBB REQUIRED)

# Compile library and programs.
ADD_SUBDIRECTORY("lib")

IF(WITH_PROGRAMS OR WITH_TESTS)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProgramUtilities.cmake)
ENDIF()

IF(WITH_PROGRAMS)
    ADD_SUBDIRECTORY("programs")
ENDIF()

IF(WITH_TESTS)
    ADD_SUBDIRECTORY("tests")
ENDIF()

# compile_commands.json for YCM.
ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
                   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
ADD_CUSTOM_TARGET(compile_commands ALL
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
