if(CMAKE_VERSION VERSION_LESS 3.26)
    message(FATAL_ERROR
        "GOB_BUILD_LUISA_RENDERER requires CMake 3.26 or newer because the pinned LuisaCompute source requires it.")
endif()

find_package(CUDAToolkit 12.1 REQUIRED)
set(GOBOT_LUISA_TARGETS_FILE "")
foreach(GOBOT_LUISA_LIBDIR lib lib64)
    set(GOBOT_LUISA_TARGETS_CANDIDATE
            "${GOB_LUISA_COMPUTE_ROOT}/${GOBOT_LUISA_LIBDIR}/cmake/LuisaCompute/LuisaComputeTargets.cmake")
    if(EXISTS "${GOBOT_LUISA_TARGETS_CANDIDATE}")
        set(GOBOT_LUISA_TARGETS_FILE "${GOBOT_LUISA_TARGETS_CANDIDATE}")
        break()
    endif()
endforeach()
if(NOT GOBOT_LUISA_TARGETS_FILE)
    message(FATAL_ERROR
        "LuisaCompute was not found under ${GOB_LUISA_COMPUTE_ROOT}.\n"
        "Build the pinned SDK first with: cmake --workflow --preset dependencies-render")
endif()
include("${GOBOT_LUISA_TARGETS_FILE}")
if(NOT TARGET luisa::compute)
    message(FATAL_ERROR "${GOBOT_LUISA_TARGETS_FILE} does not export luisa::compute.")
endif()

add_library(gobot_luisa_renderer MODULE
        cuda_gl_interop.cpp
        gaussian_splat_bridge.cpp
        luisa_renderer.cpp
        luisa_render_product.cpp
        luisa_scene_cache.cpp
        luisa_shaders.cpp
        module_entry.cpp)
target_include_directories(gobot_luisa_renderer PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}
        ${GOB_LUISA_COMPUTE_ROOT}/include/luisa/ext)
target_compile_definitions(gobot_luisa_renderer PRIVATE
        GOBOT_DISABLE_MATRIX_FMT=1
        GOBOT_LUISA_CUDA_TOOLKIT_VERSION="${CUDAToolkit_VERSION}")
target_link_libraries(gobot_luisa_renderer PRIVATE
        luisa::compute
        CUDA::cuda_driver
        Eigen3::Eigen
        RTTR::Core
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        glad
        "$<TARGET_FILE:gobot>")

if(GOB_BUILD_GSPLAT_INFERENCE)
    find_path(GOBOT_GSPLAT_INFERENCE_INCLUDE_DIR
            NAMES gsplat_inference/renderer.h
            PATHS "${GOB_GSPLAT_INFERENCE_ROOT}/include"
            NO_DEFAULT_PATH REQUIRED)
    find_library(GOBOT_GSPLAT_INFERENCE_LIBRARY
            NAMES gobot_gsplat_inference
            PATHS "${GOB_GSPLAT_INFERENCE_ROOT}/lib" "${GOB_GSPLAT_INFERENCE_ROOT}/lib64"
            NO_DEFAULT_PATH REQUIRED)
    target_include_directories(gobot_luisa_renderer PRIVATE
            "${GOBOT_GSPLAT_INFERENCE_INCLUDE_DIR}")
    target_compile_definitions(gobot_luisa_renderer PRIVATE GOBOT_HAS_GSPLAT_INFERENCE=1)
    target_link_libraries(gobot_luisa_renderer PRIVATE
            "${GOBOT_GSPLAT_INFERENCE_LIBRARY}"
            CUDA::cudart)
endif()

add_dependencies(gobot_luisa_renderer gobot)
set_target_properties(gobot_luisa_renderer PROPERTIES
        OUTPUT_NAME "gobot_luisa_renderer"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/gobot"
        BUILD_RPATH "$ORIGIN;$ORIGIN/luisa;${GOB_LUISA_COMPUTE_ROOT}/lib;${GOB_LUISA_COMPUTE_ROOT}/lib64;${GOB_LUISA_COMPUTE_ROOT}/bin"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/luisa;$ORIGIN/../nvidia/cuda_runtime/lib")
if(UNIX AND NOT APPLE)
    target_link_options(gobot_luisa_renderer PRIVATE "-Wl,--exclude-libs,ALL")
endif()
add_dependencies(gobot_editor gobot_luisa_renderer)

add_custom_command(TARGET gobot_luisa_renderer POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_FILE_DIR:gobot_luisa_renderer>/luisa"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${GOB_LUISA_COMPUTE_ROOT}/lib"
                "$<TARGET_FILE_DIR:gobot_luisa_renderer>/luisa"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${GOB_LUISA_COMPUTE_ROOT}/bin"
                "$<TARGET_FILE_DIR:gobot_luisa_renderer>/luisa"
        VERBATIM)

install(TARGETS gobot_luisa_renderer
        LIBRARY DESTINATION "gobot" COMPONENT python)
install(DIRECTORY "${GOB_LUISA_COMPUTE_ROOT}/lib/"
        DESTINATION "gobot/luisa" COMPONENT python
        FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
install(DIRECTORY "${GOB_LUISA_COMPUTE_ROOT}/bin/"
        DESTINATION "gobot/luisa" COMPONENT python
        FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
install(PROGRAMS "${GOB_LUISA_COMPUTE_ROOT}/bin/luisa_nvrtc"
        DESTINATION "gobot/luisa" COMPONENT python)
if(GOB_BUILD_GSPLAT_INFERENCE)
    install(FILES
            "${GOB_GSPLAT_INFERENCE_ROOT}/share/gobot_gsplat_inference/LICENSE"
            "${GOB_GSPLAT_INFERENCE_ROOT}/share/gobot_gsplat_inference/NOTICE"
            DESTINATION "gobot/licenses/gsplat_inference" COMPONENT python)
endif()
