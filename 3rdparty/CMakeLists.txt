add_subdirectory(rttr)
add_subdirectory(sdl2)

set(GLI_TEST_ENABLE OFF)
add_subdirectory(gli)


add_subdirectory(basis_univesal)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb/)

set(GLAD_SOURCES glad/src/gl.c glad/src/vulkan.c)
if(WIN32)
    set(GLAD_SOURCES ${GLAD_SOURCES} glad/src/wgl.c)
endif()
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

add_library(imgui-gobot INTERFACE)
target_sources(imgui-gobot INTERFACE
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
target_compile_definitions(imgui-gobot INTERFACE IMGUI_USER_CONFIG="${IMGUI_USER_CONFIG}")

target_include_directories(imgui-gobot INTERFACE imgui imgui/backends imgui_extension imgui/misc/cpp/)
target_link_libraries(imgui-gobot INTERFACE glad)


find_package(OpenGL REQUIRED)
add_executable(imgui_sdl2_opengl3_example imgui/examples/example_sdl2_opengl3/main.cpp)
target_link_libraries(imgui_sdl2_opengl3_example imgui ${CONAN_LIBS} glad OpenGL::GL)

add_executable(binary_to_compressed_c imgui/misc/fonts/binary_to_compressed_c.cpp)


add_subdirectory(meshoptimizer)