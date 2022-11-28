#https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/testing/googletest.html
find_package(gtest REQUIRED)

if(GTEST_FOUND)
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
else()
    message(ERROR "gtest was not configured properly")
endif()

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            EXTRA_ARGS --gtest_color=yes
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(example_add example_add.cpp)
