add_subdirectory(rttr)
add_subdirectory(intrusive_ptr)

add_subdirectory(glfw)
add_subdirectory(glad)

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
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_glfw.h
        )


target_link_libraries(imgui PUBLIC glfw glad)

target_include_directories(imgui PUBLIC imgui imgui/backends)

find_package(OpenGL REQUIRED)
add_executable(imgui_glfw_opengl3_example imgui/examples/example_glfw_opengl3/main.cpp)
target_link_libraries(imgui_glfw_opengl3_example imgui OpenGL::GL)