#https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/testing/googletest.html

add_subdirectory("${PROJECT_SOURCE_DIR}/3rdparty/googletest" "3rdparty/googletest")

mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
message(STATUS "gtest configured correctly")

add_library(gobot_test_support
        support/test_main.cpp
        support/test_render_backend.cpp)
target_compile_options(gobot_test_support PUBLIC "${GOBOT_COMPILE_OPTIONS}")
target_include_directories(gobot_test_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/support")
target_link_libraries(gobot_test_support PUBLIC gtest gmock gobot)

macro(package_add_test TESTNAME FILES)
    add_executable(${TESTNAME} ${FILES} ${ARGN})
    target_compile_options(${TESTNAME} PUBLIC "${GOBOT_COMPILE_OPTIONS}")
    target_link_libraries(${TESTNAME} gtest gmock)
    if (MSVC)
        target_link_libraries(${TESTNAME} -Wl,--whole-archive gobot_test_support)
    else ()
        target_link_libraries(${TESTNAME} -Wl,--no-as-needed gobot_test_support)
    endif ()
    
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            EXTRA_ARGS --gtest_color=yes
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_ref_counted core/test_ref_counted.cpp )
package_add_test(test_registration core/test_registration.cpp )
package_add_test(test_project_setting core/test_project_setting.cpp )
package_add_test(test_resource core/io/test_resource.cpp)
package_add_test(test_matrix core/math/test_matrix.cpp)
package_add_test(test_geometry core/math/test_geometry.cpp)
package_add_test(test_string_utils core/test_string_utils.cpp)
package_add_test(test_input core/test_input.cpp)
package_add_test(test_object_db core/test_object_db.cpp)
package_add_test(test_variant_serializer core/io/test_variant_serializer.cpp)
package_add_test(test_resource_format_scene core/io/test_resource_format_scene.cpp)
package_add_test(test_resource_format_mesh core/io/test_resource_format_mesh.cpp)
package_add_test(test_resource_format_usd core/io/test_resource_format_usd.cpp)
package_add_test(test_resource_format_mjcf core/io/test_resource_format_mjcf.cpp)
package_add_test(test_resource_format_urdf core/io/test_resource_format_urdf.cpp)
package_add_test(test_rid core/test_rid.cpp)

package_add_test(test_sdl_window core/window/test_sdl_window.cpp)

package_add_test(test_node_path scene/test_node_path.cpp)
package_add_test(test_box_mesh scene/test_primitive_mesh.cpp)
package_add_test(test_node scene/test_node.cpp)
package_add_test(test_node_complex scene/test_node_complex.cpp)
package_add_test(test_node_3d scene/test_node_3d.cpp)
package_add_test(test_scene_command scene/test_scene_command.cpp)
package_add_test(test_mesh_instance_3d scene/test_mesh_instance_3d.cpp)
package_add_test(test_collision_shape_3d scene/test_collision_shape_3d.cpp)
package_add_test(test_terrain_3d scene/test_terrain_3d.cpp)
package_add_test(test_node_creation_registry scene/test_node_creation_registry.cpp)
package_add_test(test_robot_nodes scene/test_robot_nodes.cpp)
package_add_test(test_resource_creation_registry scene/test_resource_creation_registry.cpp)
package_add_test(test_packed_scene scene/test_packed_scene.cpp)
package_add_test(test_edited_scene
        editor/test_edited_scene.cpp
        ${PROJECT_SOURCE_DIR}/src/gobot/editor/edited_scene.cpp)
package_add_test(test_editor_type_icons
        editor/test_type_icons.cpp
        ${PROJECT_SOURCE_DIR}/src/gobot/editor/imgui/type_icons.cpp)
target_include_directories(test_editor_type_icons PRIVATE
        ${PROJECT_SOURCE_DIR}/3rdparty/imgui_extension)
add_executable(test_python_panel
        editor/test_python_panel.cpp)
target_compile_options(test_python_panel PUBLIC "${GOBOT_COMPILE_OPTIONS}")
target_compile_definitions(test_python_panel
        PRIVATE
            GOBOT_TEST_BUILD_PYTHON_DIR="${CMAKE_BINARY_DIR}/python"
            GOBOT_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
target_include_directories(test_python_panel PRIVATE
        ${PROJECT_SOURCE_DIR}/3rdparty/ImGuiColorTextEdit
        ${PROJECT_SOURCE_DIR}/3rdparty/imgui
        ${PROJECT_SOURCE_DIR}/3rdparty/imgui_extension)
target_link_libraries(test_python_panel gtest gmock gobot_editor_runtime)
if (MSVC)
    target_link_libraries(test_python_panel -Wl,--whole-archive gobot_test_support)
else ()
    target_link_libraries(test_python_panel -Wl,--no-as-needed gobot_test_support)
endif ()
gtest_discover_tests(test_python_panel
        EXTRA_ARGS --gtest_color=yes
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_target_properties(test_python_panel PROPERTIES FOLDER tests)
add_executable(test_scene_play_session
        editor/test_scene_play_session.cpp
        ${PROJECT_SOURCE_DIR}/src/gobot/editor/scene_play_session.cpp
        ${PROJECT_SOURCE_DIR}/src/gobot/editor/python_script_sync.cpp)
target_compile_options(test_scene_play_session PUBLIC "${GOBOT_COMPILE_OPTIONS}")
target_compile_definitions(test_scene_play_session
        PRIVATE
            GOBOT_TEST_BUILD_PYTHON_DIR="${CMAKE_BINARY_DIR}/python"
            GOBOT_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
target_link_libraries(test_scene_play_session gtest gmock)
if (MSVC)
    target_link_libraries(test_scene_play_session -Wl,--whole-archive gobot_test_support)
else ()
    target_link_libraries(test_scene_play_session -Wl,--no-as-needed gobot_test_support)
endif ()
add_dependencies(test_scene_play_session gobot_python)
gtest_discover_tests(test_scene_play_session
        EXTRA_ARGS --gtest_color=yes
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_target_properties(test_scene_play_session PROPERTIES FOLDER tests)
package_add_test(test_physics_server physics/test_physics_server.cpp)
package_add_test(test_joint_controller physics/test_joint_controller.cpp)
package_add_test(test_simulation_server simulation/test_simulation_server.cpp)
package_add_test(test_locomotion_command_runtime simulation/test_locomotion_command_runtime.cpp)
package_add_test(test_locomotion_batch_runtime simulation/test_locomotion_batch_runtime.cpp)

set(GOBOT_PYTHON_TEST_PATH "${CMAKE_BINARY_DIR}/python:${CMAKE_SOURCE_DIR}")
if(Python3_SITELIB)
    string(APPEND GOBOT_PYTHON_TEST_PATH ":${Python3_SITELIB}")
endif()
if(Python3_SITEARCH AND NOT Python3_SITEARCH STREQUAL Python3_SITELIB)
    string(APPEND GOBOT_PYTHON_TEST_PATH ":${Python3_SITEARCH}")
endif()
set(GOBOT_PYTHON_TEST_ENV
        "HOME=/tmp/gobot-test-home;PYTHONNOUSERSITE=1;PYTHONPATH=${GOBOT_PYTHON_TEST_PATH}")
string(APPEND GOBOT_PYTHON_TEST_ENV
        ";GOBOT_TEST_BUILD_PYTHON_DIR=${CMAKE_BINARY_DIR}/python")
if(GOBOT_PYTHON_LIBRARY_DIR)
    string(APPEND GOBOT_PYTHON_TEST_ENV
            ";LD_LIBRARY_PATH=${GOBOT_PYTHON_LIBRARY_DIR}:${CMAKE_BINARY_DIR}/python/gobot")
endif()

add_test(
        NAME test_architecture_dependencies
        COMMAND ${Python3_EXECUTABLE} -S
                ${CMAKE_CURRENT_SOURCE_DIR}/architecture/test_dependency_boundaries.py
)
set_tests_properties(test_architecture_dependencies
        PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_test(
        NAME test_python_bindings_smoke
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/python/test_python_bindings_smoke.py
)
set_tests_properties(test_python_bindings_smoke
        PROPERTIES
            ENVIRONMENT "${GOBOT_PYTHON_TEST_ENV}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
add_test(
        NAME test_python_policy_manifest
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/python/test_policy_manifest.py
)
set_tests_properties(test_python_policy_manifest
        PROPERTIES
            ENVIRONMENT "${GOBOT_PYTHON_TEST_ENV}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
add_test(
        NAME test_python_terrain_generation_scene
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/python/test_terrain_generation_scene.py
)
set_tests_properties(test_python_terrain_generation_scene
        PROPERTIES
            ENVIRONMENT "${GOBOT_PYTHON_TEST_ENV}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
add_test(
        NAME test_velocity_env
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/python/test_velocity_env.py
)
set_tests_properties(test_velocity_env
        PROPERTIES
            ENVIRONMENT "${GOBOT_PYTHON_TEST_ENV}"
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if(GOB_BUILD_MUJOCO AND GOBOT_MUJOCO_TARGET)
    add_test(
            NAME test_python_go1_mujoco_sensor_smoke
            COMMAND ${Python3_EXECUTABLE} -S
                    ${CMAKE_CURRENT_SOURCE_DIR}/python/smoke_real_scene.py
                    --project examples/go1
                    --scene res://go1_scene.jscn
                    --backend mujoco
                    --steps 600
                    --expect-go1-stand
                    --expect-go1-sensors
    )
    set_tests_properties(test_python_go1_mujoco_sensor_smoke
            PROPERTIES
                ENVIRONMENT "${GOBOT_PYTHON_TEST_ENV}"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
