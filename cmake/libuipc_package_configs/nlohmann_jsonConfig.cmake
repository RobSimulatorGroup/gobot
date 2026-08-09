if(NOT TARGET nlohmann_json::nlohmann_json)
    set(nlohmann_json_FOUND FALSE)
    set(nlohmann_json_NOT_FOUND_MESSAGE
        "Gobot did not configure its pinned nlohmann_json target")
    return()
endif()
set(nlohmann_json_FOUND TRUE)
