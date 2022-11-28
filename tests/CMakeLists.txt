#https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/testing/googletest.html
find_package(gtest CONFIG REQUIRED)

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
    target_link_libraries(${TESTNAME} gtest gmock gtest_main gobot)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            EXTRA_ARGS --gtest_color=yes
            WORKING_DIRECTORY ${PROJECT_DIR}
            )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_ref_counted core/test_ref_counted.cpp )
