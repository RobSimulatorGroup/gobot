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
pybind11_add_module(_coal_native MODULE NO_EXTRAS planning_collision.cpp)
add_dependencies(_coal_native gobot_coal_sdk)
target_include_directories(_coal_native SYSTEM PRIVATE "${coal_source}/include" "${coal_build}/include")
target_link_libraries(_coal_native PRIVATE Eigen3::Eigen "${coal_lib}")
set_target_properties(_coal_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/gobot"
    INSTALL_RPATH "$ORIGIN")
install(TARGETS _coal_native LIBRARY DESTINATION gobot COMPONENT python)
install(DIRECTORY "${coal_build}/lib/" DESTINATION gobot COMPONENT python FILES_MATCHING PATTERN "libcoal.so*")
