if(NOT TARGET magic_enum::magic_enum)
    set(magic_enum_FOUND FALSE)
    set(magic_enum_NOT_FOUND_MESSAGE
        "Gobot did not configure its pinned magic_enum target")
    return()
endif()
set(magic_enum_FOUND TRUE)
