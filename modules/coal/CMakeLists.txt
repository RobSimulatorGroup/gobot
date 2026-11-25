# Isolate Coal's upstream build options and dependency targets from the engine.
include(ExternalProject)
set(coal_source "${GOBOT_ROOT_DIR}/3rdparty/coal")
if(NOT EXISTS "${coal_source}/cmake/base.cmake")
    message(FATAL_ERROR "Initialize Coal with git submodule update --init --recursive 3rdparty/coal")
endif()
set(coal_build "${CMAKE_CURRENT_BINARY_DIR}/upstream")
set(coal_lib "${coal_build}/lib/libcoal.so")
ExternalProject_Add(gobot_coal_sdk SOURCE_DIR "${coal_source}" BINARY_DIR "${coal_build}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_STANDARD=17 -DBUILD_TESTING=OFF -DBUILD_PYTHON_INTERFACE=OFF
               -DCOAL_HAS_QHULL=OFF -DCOAL_TURN_ASSERT_INTO_EXCEPTION=ON
               -DCMAKE_DISABLE_FIND_PACKAGE_octomap=ON -DINSTALL_DOCUMENTATION=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target coal
    INSTALL_COMMAND "" BUILD_ALWAYS TRUE BUILD_BYPRODUCTS "${coal_lib}")

if(NOT TARGET TBB::tbb)
    find_package(TBB CONFIG QUIET)
endif()

# Standalone C++ motion planning library (no Python/pybind11 dependency) for C++ engine,
# editor tools, C++ unit tests, and the thin Python binding module.
add_library(gobot_planning SHARED planning_lib.cpp)
add_dependencies(gobot_planning gobot_coal_sdk)
target_include_directories(gobot_planning PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(gobot_planning SYSTEM PUBLIC "${coal_source}/include" "${coal_build}/include")
target_link_libraries(gobot_planning PUBLIC Eigen3::Eigen "${coal_lib}")
if(TARGET TBB::tbb)
    target_link_libraries(gobot_planning PUBLIC TBB::tbb)
    target_compile_definitions(gobot_planning PUBLIC GOBOT_PLANNING_HAS_TBB=1)
endif()
set_target_properties(gobot_planning PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
    BUILD_RPATH "${coal_build}/lib"
    INSTALL_RPATH "$ORIGIN")

# Thin pybind11 wrapper over gobot_planning.
pybind11_add_module(_coal_native MODULE NO_EXTRAS planning_collision.cpp)
target_link_libraries(_coal_native PRIVATE gobot_planning)
set_target_properties(_coal_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
    INSTALL_RPATH "$ORIGIN")

install(TARGETS gobot_planning _coal_native LIBRARY DESTINATION gobot COMPONENT python)
install(DIRECTORY "${coal_build}/lib/" DESTINATION gobot COMPONENT python FILES_MATCHING PATTERN "libcoal.so*")
