FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(GMock REQUIRED)

SET(MOCK_FILES mocks/scene_loader_mock.h
               mocks/scene_builder_mock.h
               mocks/mesh_loader_mock.h
               mocks/camera_loader_mock.h
               mocks/xform_loader_mock.h)

FUNCTION(EASY_CREATE_TEST NM)
    ADD_EXECUTABLE(${NM} ${NM}.cpp test_common.cpp test_common.h ${MOCK_FILES})
    SETUP_EXECUTABLE(${NM})
    TARGET_INCLUDE_DIRECTORIES(${NM} SYSTEM PUBLIC ${GMOCK_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${NM} ${GMOCK_LIBRARIES} ${GTEST_LIBRARIES})
    ADD_TEST(NAME "${NM}_TEST"
             COMMAND ${NM}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
ENDFUNCTION()

#SET(TEST_CPPS mesh_loader_test.cpp
#              scene_loader_test.cpp
#              vertex_container_test.cpp
#              triangle_test.cpp
#              angle_test.cpp
#              distance_test.cpp
#              pinhole_perspective_camera_test.cpp
#              transform_test.cpp
#              rotation_test.cpp
#              xform_loader_test.cpp
#              camera_loader_test.cpp)
SET(TEST_CPPS scene_loader_test.cpp)

FOREACH(FNAME ${TEST_CPPS})
    GET_FILENAME_COMPONENT(BASE ${FNAME} NAME_WE)
    EASY_CREATE_TEST(${BASE})
ENDFOREACH()
