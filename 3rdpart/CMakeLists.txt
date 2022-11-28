add_subdirectory(rttr)

add_library(intrusive_ptr INTERFACE)
target_include_directories(intrusive_ptr INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/intrusive_ptr>
        $<INSTALL_INTERFACE:include/intrusive_ptr>
        )
