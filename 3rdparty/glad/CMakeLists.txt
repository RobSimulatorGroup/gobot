add_library(
        glad
        src/glad.c
        include/glad/glad.h
        include/KHR/khrplatform.h
)

target_include_directories(glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
