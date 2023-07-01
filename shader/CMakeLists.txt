set(TARGET_NAME ${SHADER_COMPILE_TARGET})

file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.geom"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.tesc"
        "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.tese")

file(GLOB_RECURSE SHADER_INCLUDE_FILES
        CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/glsl/*.hpp")

SOURCE_GROUP("shaders" FILES ${SHADER_FILES})

set(GENERATED_SHADER_FOLDER "generated")
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${GENERATED_SHADER_FOLDER})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${GENERATED_SHADER_FOLDER})
endif()


add_executable(GlslCppSaver glsl_hpp_generator.cpp)
target_link_libraries(GlslCppSaver  PUBLIC   ${CONAN_LIBS})

set(working_dir "${CMAKE_CURRENT_SOURCE_DIR}")

set(ALL_GENERATED_CPP_FILES "")


foreach(SHADER ${SHADER_FILES})
    # Prepare a header name and a global variable for this shader
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    string(REPLACE "." "_" HEADER_NAME ${SHADER_NAME})
    string(TOUPPER ${HEADER_NAME} GLOBAL_SHADER_VAR)

    set(CPP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${GENERATED_SHADER_FOLDER}/glsl_shader_hpp/${HEADER_NAME}.hpp")

    add_custom_command(
            OUTPUT ${CPP_FILE}
            COMMAND GlslCppSaver ${GLOBAL_SHADER_VAR} ${CPP_FILE} ${SHADER}
            DEPENDS ${SHADER} ${SHADER_INCLUDE_FILES}
            WORKING_DIRECTORY "${working_dir}"
    )

    list(APPEND ALL_GENERATED_CPP_FILES ${CPP_FILE})

endforeach()

add_custom_target(${TARGET_NAME} DEPENDS "${ALL_GENERATED_CPP_FILES}" SOURCES ${SHADERS})

