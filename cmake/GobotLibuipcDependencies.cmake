include_guard(GLOBAL)

set(GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR
    "${CMAKE_CURRENT_LIST_DIR}/libuipc_package_configs")

foreach(GOBOT_LIBUIPC_EXISTING_TARGET
        Eigen3::Eigen
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        magic_enum::magic_enum)
    if(NOT TARGET "${GOBOT_LIBUIPC_EXISTING_TARGET}")
        message(FATAL_ERROR
            "libuipc requires the existing Gobot target ${GOBOT_LIBUIPC_EXISTING_TARGET}")
    endif()
endforeach()

find_package(TBB REQUIRED)
find_package(urdfdom CONFIG REQUIRED)
foreach(GOBOT_LIBUIPC_URDFDOM_TARGET
        urdfdom::urdfdom_model
        urdfdom::urdfdom_world
        urdfdom::urdfdom_sensor)
    if(NOT TARGET "${GOBOT_LIBUIPC_URDFDOM_TARGET}")
        message(FATAL_ERROR
            "libuipc requires the system target ${GOBOT_LIBUIPC_URDFDOM_TARGET}")
    endif()
endforeach()
if(NOT TARGET urdfdom::urdf_parser)
    # Ubuntu 22.04 exports the parser implementation through these three
    # libraries but predates urdfdom's equivalent interface target.
    add_library(urdfdom::urdf_parser INTERFACE IMPORTED GLOBAL)
    set_target_properties(urdfdom::urdf_parser PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "urdfdom::urdfdom_model;urdfdom::urdfdom_sensor;urdfdom::urdfdom_world")
endif()

# These revisions match the dependency versions selected by the pinned
# libuipc source. Header-only dependencies are exposed through the exact target
# names expected by libuipc without installing a second package manager.
CPMAddPackage(
    NAME gobot_libuipc_cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG 539a5be8359c4330b3f88ed1821f32bb5c89f5f6
    DOWNLOAD_ONLY YES)
set(GOBOT_LIBUIPC_GENERATED_INCLUDE_DIR
    "${CMAKE_BINARY_DIR}/libuipc_include")
file(MAKE_DIRECTORY "${GOBOT_LIBUIPC_GENERATED_INCLUDE_DIR}")
if(NOT EXISTS "${GOBOT_LIBUIPC_GENERATED_INCLUDE_DIR}/cppitertools")
    file(CREATE_LINK
        "${gobot_libuipc_cppitertools_SOURCE_DIR}"
        "${GOBOT_LIBUIPC_GENERATED_INCLUDE_DIR}/cppitertools"
        SYMBOLIC)
endif()
if(NOT TARGET cppitertools)
    add_library(cppitertools INTERFACE)
    target_include_directories(cppitertools INTERFACE
        "${GOBOT_LIBUIPC_GENERATED_INCLUDE_DIR}")
    target_compile_features(cppitertools INTERFACE cxx_std_17)
    add_library(cppitertools::cppitertools ALIAS cppitertools)
endif()

CPMAddPackage(
    NAME gobot_libuipc_dylib
    GITHUB_REPOSITORY martin-olivier/dylib
    GIT_TAG fe48b1b65c91d21973bd8b81704e09d059c54291
    DOWNLOAD_ONLY YES)
set(DYLIB_INCLUDE_DIRS
    "${gobot_libuipc_dylib_SOURCE_DIR}/include"
    CACHE PATH "Pinned dylib headers used by libuipc" FORCE)

CPMAddPackage(
    NAME gobot_libuipc_tinygltf
    GITHUB_REPOSITORY syoyo/tinygltf
    GIT_TAG 9aa1c3c393b4d1b6a89323767f46f9dc0de3cb2f
    DOWNLOAD_ONLY YES)
set(TINYGLTF_INCLUDE_DIRS
    "${gobot_libuipc_tinygltf_SOURCE_DIR}"
    CACHE PATH "Pinned tinygltf headers used by libuipc" FORCE)

CPMAddPackage(
    NAME gobot_libuipc_libigl
    GITHUB_REPOSITORY libigl/libigl
    GIT_TAG 40e7900ccbd767f1f360e0eb10f0f1a6432e0993
    DOWNLOAD_ONLY YES)
if(NOT TARGET gobot_libuipc_igl_core)
    find_package(Threads REQUIRED)
    add_library(gobot_libuipc_igl_core INTERFACE)
    target_include_directories(gobot_libuipc_igl_core INTERFACE
        "${gobot_libuipc_libigl_SOURCE_DIR}/include")
    target_link_libraries(gobot_libuipc_igl_core INTERFACE
        Eigen3::Eigen Threads::Threads)
    add_library(igl::igl_core ALIAS gobot_libuipc_igl_core)
endif()

CPMAddPackage(
    NAME gobot_libuipc_octree
    GITHUB_REPOSITORY SpiriMirror/Octree
    GIT_TAG e1c8300b4d43cad46566b1d6a105b48cbf89dd13
    DOWNLOAD_ONLY YES)
if(NOT TARGET gobot_libuipc_octree_target)
    add_library(gobot_libuipc_octree_target INTERFACE)
    target_include_directories(gobot_libuipc_octree_target INTERFACE
        "${gobot_libuipc_octree_SOURCE_DIR}")
    target_compile_features(gobot_libuipc_octree_target INTERFACE cxx_std_20)
    add_library(Octree::Octree ALIAS gobot_libuipc_octree_target)
endif()

set(CPPTRACE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(CPPTRACE_GET_SYMBOLS_WITH_NOTHING ON CACHE BOOL "" FORCE)
set(CPPTRACE_UNWIND_WITH_EXECINFO ON CACHE BOOL "" FORCE)
set(CPPTRACE_DEMANGLE_WITH_CXXABI ON CACHE BOOL "" FORCE)
set(CPPTRACE_PROVIDE_EXPORT_SET OFF CACHE BOOL "" FORCE)
CPMAddPackage(
    NAME cpptrace
    GITHUB_REPOSITORY jeremy-rifkin/cpptrace
    GIT_TAG ce639ebfcec47a7c74233b4bab50017cb34e615b
    OPTIONS
        "CPPTRACE_BUILD_SHARED OFF"
        "CPPTRACE_BUILD_TESTING OFF"
        "CPPTRACE_BUILD_TOOLS OFF"
        "CPPTRACE_BUILD_BENCHMARK OFF"
        "CPPTRACE_GET_SYMBOLS_WITH_NOTHING ON"
        "CPPTRACE_UNWIND_WITH_EXECINFO ON"
        "CPPTRACE_DEMANGLE_WITH_CXXABI ON"
        "CPPTRACE_PROVIDE_EXPORT_SET OFF")
if(NOT TARGET cpptrace::cpptrace)
    message(FATAL_ERROR "The pinned cpptrace source did not define cpptrace::cpptrace")
endif()

# Redirect libuipc's CONFIG-mode lookups to validation-only package files.
# All corresponding targets above are already pinned and available.
set(fmt_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(spdlog_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(nlohmann_json_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(magic_enum_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(cppitertools_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(cpptrace_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(Octree_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")
set(libigl_DIR "${GOBOT_LIBUIPC_PACKAGE_CONFIG_DIR}")

gobot_install_license_files(
    "${gobot_libuipc_cppitertools_SOURCE_DIR}" "cppitertools"
    python libuipc_runtime)
gobot_install_license_files("${gobot_libuipc_dylib_SOURCE_DIR}" "dylib"
    python libuipc_runtime)
gobot_install_license_files("${gobot_libuipc_tinygltf_SOURCE_DIR}" "tinygltf"
    python libuipc_runtime)
gobot_install_license_files("${gobot_libuipc_libigl_SOURCE_DIR}" "libigl"
    python libuipc_runtime)
gobot_install_license_files("${gobot_libuipc_octree_SOURCE_DIR}" "Octree"
    python libuipc_runtime)
gobot_install_license_files("${cpptrace_SOURCE_DIR}" "cpptrace"
    python libuipc_runtime)
gobot_install_license_files("${GOBOT_LIBUIPC_SOURCE_DIR}/external/GKlib" "GKlib"
    python libuipc_runtime)
gobot_install_license_files("${GOBOT_LIBUIPC_SOURCE_DIR}/external/METIS" "METIS"
    python libuipc_runtime)
gobot_install_license_files("${GOBOT_LIBUIPC_SOURCE_DIR}/external/muda" "muda"
    python libuipc_runtime)

# These Gobot dependencies are linked statically or instantiated into the
# libuipc shared objects. Their normal Python-component rules remain above;
# repeat only the license payload for the standalone runtime bundle.
gobot_install_license_files("${fmt_SOURCE_DIR}" "fmt" libuipc_runtime)
gobot_install_license_files("${spdlog_SOURCE_DIR}" "spdlog" libuipc_runtime)
gobot_install_license_files("${nlohmann_json_SOURCE_DIR}" "nlohmann_json"
    libuipc_runtime)
gobot_install_license_files("${magic_enum_SOURCE_DIR}" "magic_enum"
    libuipc_runtime)
