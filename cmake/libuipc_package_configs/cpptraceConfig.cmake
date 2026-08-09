if(NOT TARGET cpptrace::cpptrace)
    set(cpptrace_FOUND FALSE)
    set(cpptrace_NOT_FOUND_MESSAGE
        "Gobot did not configure its pinned cpptrace target")
    return()
endif()
set(cpptrace_FOUND TRUE)
