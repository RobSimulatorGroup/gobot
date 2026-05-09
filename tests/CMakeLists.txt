#https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/testing/googletest.html

add_subdirectory("${PROJECT_SOURCE_DIR}/3rdparty/googletest" "3rdparty/googletest")

mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
message(STATUS "gtest configured correctly")


macro(package_add_test TESTNAME FILES)
    add_executable(${TESTNAME} ${FILES} ${ARGN})
    target_compile_options(${TESTNAME} PUBLIC "${GOBOT_COMPILE_OPTIONS}")
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    if (MSVC)
        target_link_libraries(${TESTNAME} -Wl,--whole-archive gobot)
    else ()
        target_link_libraries(${TESTNAME} -Wl,--no-as-needed gobot)
    endif ()
    
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            EXTRA_ARGS --gtest_color=yes
            WORKING_DIRECTORY ${PROJECT_DIR}
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
package_add_test(test_object_db core/test_object_db.cpp)
package_add_test(test_variant_serializer core/io/test_variant_serializer.cpp)
package_add_test(test_resource_format_scene core/io/test_resource_format_scene.cpp)
package_add_test(test_resource_format_mesh core/io/test_resource_format_mesh.cpp)
package_add_test(test_resource_format_usd core/io/test_resource_format_usd.cpp)
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
package_add_test(test_node_creation_registry scene/test_node_creation_registry.cpp)
package_add_test(test_robot_nodes scene/test_robot_nodes.cpp)
package_add_test(test_resource_creation_registry scene/test_resource_creation_registry.cpp)
package_add_test(test_packed_scene scene/test_packed_scene.cpp)
package_add_test(test_edited_scene
        editor/test_edited_scene.cpp
        ${PROJECT_SOURCE_DIR}/src/gobot/editor/edited_scene.cpp)
package_add_test(test_physics_server physics/test_physics_server.cpp)
package_add_test(test_joint_controller physics/test_joint_controller.cpp)
package_add_test(test_simulation_server simulation/test_simulation_server.cpp)
package_add_test(test_rl_environment simulation/test_rl_environment.cpp)

if (GOB_BUILD_PYTHON_BINDINGS)
    add_test(
            NAME test_python_bindings_smoke
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_python_bindings_smoke.py
    )
    set_tests_properties(test_python_bindings_smoke
            PROPERTIES
                ENVIRONMENT "PYTHONNOUSERSITE=1;PYTHONPATH=${CMAKE_BINARY_DIR}/python"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
