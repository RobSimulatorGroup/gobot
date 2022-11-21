CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG release-1.12.1
        VERSION 1.12.1
        OPTIONS
        "INSTALL_GTEST OFF"
        "gtest_force_shared_crt ON"
        "BUILD_GMOCK ON"
)
if(googletest_ADDED)
    message(STATUS "gtest configured correctly")
else()
    message(ERROR "gtest was not configured properly")
endif()

add_executable(
        unit_tests
        example_add.cpp
)

target_link_libraries(unit_tests
        PRIVATE
        gtest_main
        )

# automatic discovery of unit tests
include(GoogleTest)
gtest_discover_tests(unit_tests
        PROPERTIES
        LABELS "unit"
        DISCOVERY_TIMEOUT  # how long to wait (in seconds) before crashing
        240
        )
