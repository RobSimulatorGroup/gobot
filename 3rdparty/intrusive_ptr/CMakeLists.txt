project(intrusive_ptr)

add_library(intrusive_ptr INTERFACE)
target_include_directories(intrusive_ptr INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
