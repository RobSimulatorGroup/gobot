SET(BUILD_SHARED_LIBS OFF CACHE BOOL "BUILD_SHARED_LIBS" FORCE)
SET(BUILD_STATIC_LIBS ON CACHE BOOL "BUILD_STATIC_LIBS" FORCE)
SET(POSITION_INDEPENDENT_CODE ON CACHE BOOL "POSITION_INDEPENDENT_CODE" FORCE)

# SDL2
CPMAddPackage(
        NAME SDL2
        GITHUB_REPOSITORY libsdl-org/SDL
        GIT_TAG release-2.26.4
        OPTIONS
        "SDL2_DISABLE_INSTALL ON"
        "SDL_SHARED OFF"
        "SDL_STATIC ON"
        "SDL_STATIC_PIC ON"
        "SDL_WERROR OFF"
)
if (SDL2_ADDED)
    add_library(SDL2::SDL2 ALIAS SDL2-static)
    file(GLOB SDL2_HEADERS "${SDL2_SOURCE_DIR}/include/*.h")

    # And add the directory where headers have been copied as an interface include dir
    target_include_directories(SDL2-static INTERFACE "${SDL2_SOURCE_DIR}/include")
    set_property(TARGET SDL2-static PROPERTY C_STANDARD 99)

    set (SDL2_INCLUDE_DIR ${SDL2_SOURCE_DIR}/include)
    include_directories(${SDL2_INCLUDE_DIR})
endif()


# SDL_image
CPMAddPackage(GITHUB_REPOSITORY libsdl-org/SDL_image
        GIT_TAG release-2.6.2
        OPTIONS
        "SDL2IMAGE_INSTALL OFF"
        "SDL2IMAGE_SAMPLES OFF"
        "SDL2IMAGE_VENDORED ON"
        "SDL2IMAGE_BUILD_SHARED_LIBS OFF"
        )
find_package(SDL_image REQUIRED)
include_directories(${SDL_image_SOURCE_DIR})