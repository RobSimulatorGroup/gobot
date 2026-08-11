find_package(CUDAToolkit REQUIRED)

add_library(gobot_libuipc_solver MODULE
        libuipc_solver_module.cpp)
target_compile_features(gobot_libuipc_solver PRIVATE cxx_std_20)
target_include_directories(gobot_libuipc_solver PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(gobot_libuipc_solver PRIVATE
        GOBOT_LIBUIPC_BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/$<CONFIG>/bin"
        GOBOT_LIBUIPC_REVISION="619a5412cf958eb59cf3c9cebc9a8e8e9625ebd3")
target_link_libraries(gobot_libuipc_solver PRIVATE
        uipc::uipc
        CUDA::cudart)
add_dependencies(gobot_libuipc_solver uipc_backends)
set_target_properties(gobot_libuipc_solver PROPERTIES
        OUTPUT_NAME "gobot_libuipc_solver"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/gobot"
        BUILD_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/$<CONFIG>/bin"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/libuipc/$<CONFIG>/bin")
add_dependencies(gobot_editor gobot_libuipc_solver)

foreach(GOBOT_LIBUIPC_INSTALL_COMPONENT python libuipc_runtime)
    install(TARGETS gobot_libuipc_solver
            LIBRARY DESTINATION "gobot"
            COMPONENT "${GOBOT_LIBUIPC_INSTALL_COMPONENT}")
endforeach()

# libuipc's own install rules use the default CMake component. scikit-build
# installs only Gobot's `python` component, so explicitly include the private
# runtime libraries beside the module without enabling libuipc's pyuipc wheel.
get_property(GOBOT_LIBUIPC_BACKEND_TARGETS
        GLOBAL PROPERTY UIPC_BACKEND_TARGETS)
set(GOBOT_LIBUIPC_RUNTIME_TARGETS
        uipc_core
        uipc_geometry
        uipc_constitution
        uipc_io
        uipc_sanity_check
        ${GOBOT_LIBUIPC_BACKEND_TARGETS})
set(GOBOT_LIBUIPC_RUNTIME_INSTALL_DIR
        "gobot/libuipc/$<CONFIG>/bin")
foreach(GOBOT_LIBUIPC_INSTALL_COMPONENT python libuipc_runtime)
    install(TARGETS ${GOBOT_LIBUIPC_RUNTIME_TARGETS}
            RUNTIME DESTINATION "${GOBOT_LIBUIPC_RUNTIME_INSTALL_DIR}"
            LIBRARY DESTINATION "${GOBOT_LIBUIPC_RUNTIME_INSTALL_DIR}"
            COMPONENT "${GOBOT_LIBUIPC_INSTALL_COMPONENT}")
endforeach()

gobot_install_license_files("${PROJECT_SOURCE_DIR}/3rdparty/libuipc" "libuipc"
        python libuipc_runtime)
