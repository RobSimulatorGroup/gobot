

add_subdirectory(rttr)

CPMAddPackage(
        NAME spdlog
        GITHUB_REPOSITORY gabime/spdlog
        VERSION 1.11.0
        EXCLUDE_FROM_ALL YES)

CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.2
        EXCLUDE_FROM_ALL YES)

CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        VERSION 2.10.1
        EXCLUDE_FROM_ALL YES)

add_library(intrusive_ptr INTERFACE)
target_include_directories(intrusive_ptr INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/intrusive_ptr>
        $<INSTALL_INTERFACE:include/intrusive_ptr>  # <prefix>/include/mylib
        )
