FUNCTION(SETUP_EXECUTABLE TARGET)
    # Internal library
    TARGET_INCLUDE_DIRECTORIES(${TARGET} SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    TARGET_LINK_LIBRARIES(${TARGET} cpt)

    # Boost
    TARGET_INCLUDE_DIRECTORIES(${TARGET} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(${TARGET} Boost::system Boost::filesystem Boost::program_options)
ENDFUNCTION()
