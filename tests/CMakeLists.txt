#https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/testing/googletest.html

find_package(GTest CONFIG REQUIRED)

include(GoogleTest)
mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
set_target_properties(GTest::gtest PROPERTIES FOLDER extern)
set_target_properties(GTest::gtest_main PROPERTIES FOLDER extern)
set_target_properties(GTest::gmock PROPERTIES FOLDER extern)
set_target_properties(GTest::gmock_main PROPERTIES FOLDER extern)
message(STATUS "gtest configured correctly")


macro(package_add_test TESTNAME FILES)
    add_executable(${TESTNAME} ${FILES})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    if (MSVC)
        target_link_libraries(${TESTNAME} gobot)
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
package_add_test(test_string_utils core/test_string_utils.cpp)
package_add_test(test_variant_serializer core/io/test_variant_serializer.cpp)
package_add_test(test_resource_format_scene core/io/test_resource_format_scene.cpp)

package_add_test(test_sdl_window core/window/test_sdl_window.cpp)

package_add_test(test_node_path scene/test_node_path.cpp)
package_add_test(test_box_mesh scene/test_primitive_mesh.cpp)
package_add_test(test_node scene/test_node.cpp)
package_add_test(test_node_complex scene/test_node_complex.cpp)
package_add_test(test_node_3d scene/test_node_3d.cpp)
