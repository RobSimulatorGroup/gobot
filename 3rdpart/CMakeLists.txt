

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