add_subdirectory(rttr)

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
        imgui/backends/imgui_impl_opengl3.cpp
        imgui/backends/imgui_impl_opengl3.h
        imgui/backends/imgui_impl_sdl2.h
        imgui/backends/imgui_impl_sdl2.cpp
        )

target_include_directories(imgui PUBLIC imgui imgui/backends imgui_extension )

find_package(OpenGL REQUIRED)
add_executable(imgui_sdl2_opengl3_example imgui/examples/example_sdl2_opengl3/main.cpp)
target_link_libraries(imgui_sdl2_opengl3_example imgui ${CONAN_LIBS} OpenGL::GL)

add_executable(binary_to_compressed_c imgui/misc/fonts/binary_to_compressed_c.cpp)

set(BGFX_WITH_SDL On)
add_subdirectory(bgfx)