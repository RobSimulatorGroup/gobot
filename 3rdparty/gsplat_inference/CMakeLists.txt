cmake_minimum_required(VERSION 3.26)
project(gobot_gsplat_inference LANGUAGES CXX CUDA)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GOBOT_GSPLAT_CUDA_ARCHITECTURES
        "75-real;80-real;86-real;89-real;90-real"
        CACHE STRING "CUDA architectures for the Gobot gsplat inference library")

add_library(gobot_gsplat_inference STATIC src/renderer.cu)
target_include_directories(gobot_gsplat_inference
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(gobot_gsplat_inference PROPERTIES
        CUDA_ARCHITECTURES "${GOBOT_GSPLAT_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(gobot_gsplat_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)

install(TARGETS gobot_gsplat_inference
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/gobot_gsplat_inference)
