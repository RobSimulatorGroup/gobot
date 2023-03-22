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
        "SDL_TEST_ENABLED_BY_DEFAULT OFF"
)
if (SDL2_ADDED)
    add_library(SDL2::SDL2 ALIAS SDL2-static)


    # Create a target that copies headers at build time, when they change
    string(TOLOWER "${CMAKE_BUILD_TYPE}" lower_build_type)
    add_custom_target(sdl_copy_headers_in_build_dir
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${SDL2_BINARY_DIR}/include" "${CMAKE_BINARY_DIR}/SDLHeaders/SDL2"
            DEPENDS ${SDL_HEADERS})
    add_custom_target(sdl_copy_config_in_build_dir
            COMMAND ${CMAKE_COMMAND} -E copy "${SDL2_BINARY_DIR}/include-config-${lower_build_type}/SDL_config.h" "${CMAKE_BINARY_DIR}/SDLHeaders/SDL2/")


    # Make SDL depend from it
    add_dependencies(SDL2-static sdl_copy_headers_in_build_dir )
    add_dependencies(sdl_copy_config_in_build_dir sdl_copy_headers_in_build_dir )

    ## For using of #include <SDL2/SDL.h>
    # And add the directory where headers have been copied as an interface include dir
    target_include_directories(SDL2-static INTERFACE "${CMAKE_BINARY_DIR}/SDLHeaders")

    ## For using of #include <SDL.h>
    # And add the directory where headers have been copied as an interface include dir
    target_include_directories(SDL2-static INTERFACE "${CMAKE_BINARY_DIR}/SDLHeaders/SDL2")

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