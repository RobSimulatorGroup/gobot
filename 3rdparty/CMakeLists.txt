set(BUILD_RTTR_DYNAMIC OFF CACHE BOOL "Build the dynamic/shared version of RTTR library." FORCE)
set(BUILD_STATIC ON CACHE BOOL "Build RTTR as static library." FORCE)
add_subdirectory(rttr)
if(NOT TARGET RTTR::Core AND TARGET rttr_core_lib)
    add_library(RTTR::Core ALIAS rttr_core_lib)
endif()

set(SDL_X11_XINPUT OFF CACHE BOOL "Disable SDL XInput2 support to avoid XGenericEventCookie typedef conflicts with some system X11 headers." FORCE)
set(HAVE_XGENERICEVENT TRUE CACHE INTERNAL "System Xlib provides XGenericEventCookie.")
add_subdirectory(sdl2)

set(GLI_TEST_ENABLE OFF)
add_subdirectory(gli)


#add_subdirectory(basis_univesal)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb/)

set(GLAD_SOURCES glad/src/glad.c)
add_library(glad ${GLAD_SOURCES})
target_include_directories(glad PUBLIC glad/include)

add_library(imgui
        imgui/imgui.cpp
        imgui/imgui.h
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_internal.h
        imgui/imstb_rectpack.h
        imgui/imstb_textedit.h
        imgui/imstb_truetype.h
        imgui/misc/cpp/imgui_stdlib.h
        imgui/misc/cpp/imgui_stdlib.cpp

        imgui/backends/imgui_impl_opengl3.cpp
        imgui/backends/imgui_impl_opengl3.h
        imgui/backends/imgui_impl_sdl2.h
        imgui/backends/imgui_impl_sdl2.cpp )

target_include_directories(imgui PUBLIC imgui imgui/backends imgui_extension )
target_link_libraries(imgui PUBLIC SDL2::SDL2)

add_library(imgui-gobot STATIC
        imgui/imgui.cpp
        imgui/imgui.h
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_internal.h
        imgui/imstb_rectpack.h
        imgui/imstb_textedit.h
        imgui/imstb_truetype.h
        imgui/backends/imgui_impl_opengl3.cpp
        imgui/backends/imgui_impl_opengl3.h
        imgui/backends/imgui_impl_sdl2.h
        imgui/backends/imgui_impl_sdl2.cpp

        imgui/misc/cpp/imgui_stdlib.h
        imgui/misc/cpp/imgui_stdlib.cpp

        imgui_extension/imgui_extension/gizmos/ImGuizmo.h
        imgui_extension/imgui_extension/gizmos/ImGuizmo.cpp

        )

get_filename_component(IMGUI_USER_CONFIG "imgui_config.hpp" REALPATH BASE_DIR "${PROJECT_SOURCE_DIR}/include/gobot/editor/imgui/")
target_compile_definitions(imgui-gobot PUBLIC IMGUI_USER_CONFIG="${IMGUI_USER_CONFIG}")

target_include_directories(imgui-gobot PUBLIC imgui imgui/backends imgui_extension imgui/misc/cpp/ ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR})
target_link_libraries(imgui-gobot PUBLIC SDL2::SDL2 RTTR::Core Eigen3::Eigen fmt::fmt glad magic_enum::magic_enum)

add_executable(binary_to_compressed_c imgui/misc/fonts/binary_to_compressed_c.cpp)


add_subdirectory(meshoptimizer)
