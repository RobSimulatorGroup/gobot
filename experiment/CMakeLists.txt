

# GLAD
CPMAddPackage(
        NAME glad
        GITHUB_REPOSITORY Dav1dde/glad
        VERSION 0.1.33
        OPTIONS
        "GLAD_PROFILE core"
        "GLAD_API gl=4.5"
)

CPMAddPackage(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.3.8
        OPTIONS
        "GLFW_BUILD_TESTS Off"
        "GLFW_BUILD_EXAMPLES Off"
        "GLFW_BUILD_DOCS Off"
        "GLFW_INSTALL Off"
        "GLFW_USE_HYBRID_HPG On"
)


# Libraries
find_package(Vulkan REQUIRED)

CPMAddPackage(
        NAME ImGui
        GITHUB_REPOSITORY ocornut/imgui
        GIT_TAG master
        DOWNLOAD_ONLY YES
)

# Dear ImGui
if (ImGui_ADDED)
    add_library(DearImGui
            ${ImGui_SOURCE_DIR}/imgui.cpp ${ImGui_SOURCE_DIR}/imgui_draw.cpp
            ${ImGui_SOURCE_DIR}/imgui_tables.cpp ${ImGui_SOURCE_DIR}/imgui_widgets.cpp
            ${ImGui_SOURCE_DIR}/imgui_demo.cpp
            ${ImGui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp ${ImGui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)

    target_include_directories(DearImGui PUBLIC ${ImGui_SOURCE_DIR})
    target_link_libraries(DearImGui Vulkan::Vulkan glfw)
endif ()

file(GLOB_RECURSE EXPERIMENT_SOURCE CONFIGURE_DEPENDS "src/*.cpp")

add_executable(glfw_main ${EXPERIMENT_SOURCE} )
target_link_libraries(glfw_main
        PUBLIC
        glad
        Vulkan::Vulkan
        glfw
        gobot
        )

