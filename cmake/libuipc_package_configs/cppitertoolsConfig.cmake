if(NOT TARGET cppitertools::cppitertools)
    set(cppitertools_FOUND FALSE)
    set(cppitertools_NOT_FOUND_MESSAGE
        "Gobot did not configure its pinned cppitertools target")
    return()
endif()
set(cppitertools_FOUND TRUE)
