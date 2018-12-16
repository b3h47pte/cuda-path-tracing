FUNCTION(EASY_CREATE_TEST NM)
    ADD_EXECUTABLE(${NM} "${NM}.cpp")
    SETUP_EXECUTABLE(${NM})
    TARGET_LINK_LIBRARIES(${NM} Boost::unit_test_framework)
    ADD_TEST(NAME "${NM}_TEST"
             COMMAND ${NM}
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
ENDFUNCTION()

SET(TEST_CPPS mesh_loader_test.cpp
              scene_loader_test.cpp
              vertex_container_test.cpp)

FOREACH(FNAME ${TEST_CPPS})
    GET_FILENAME_COMPONENT(BASE ${FNAME} NAME_WE)
    EASY_CREATE_TEST(${BASE})
ENDFOREACH()