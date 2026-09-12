include_guard(GLOBAL)

function(gobot_configure_superdex)
    if(CMAKE_VERSION VERSION_LESS 3.25)
        message(FATAL_ERROR
            "GOB_BUILD_SUPERDEX requires CMake 3.25 or newer. "
            "Use the CMake executable from Gobot's project environment.")
    endif()

    if(GOB_SUPERDEX_ENABLE_CUDA)
        message(FATAL_ERROR
            "GOB_SUPERDEX_ENABLE_CUDA was requested, but the pinned SuperDex SDK "
            "does not yet export its CUDA linear solver. Build the CPU backend or "
            "update the pinned SDK fork first; CUDA requests never fall back to CPU.")
    endif()

    if(GOB_REAL_T_IS_DOUBLE)
        set(GOBOT_SUPERDEX_MOCHI_LIBRARY_NAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}mochi_physics_double${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
        set(GOBOT_SUPERDEX_MOCHI_LIBRARY_NAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}mochi_physics${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    if(GOB_SUPERDEX_ROOT)
        get_filename_component(GOBOT_SUPERDEX_INSTALL_DIR
            "${GOB_SUPERDEX_ROOT}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        find_package(SuperDexPhysics REQUIRED CONFIG
            PATHS
                "${GOBOT_SUPERDEX_INSTALL_DIR}"
                "${GOBOT_SUPERDEX_INSTALL_DIR}/lib/cmake/SuperDexPhysics"
                "${GOBOT_SUPERDEX_INSTALL_DIR}/lib64/cmake/SuperDexPhysics"
            NO_DEFAULT_PATH)
        get_target_property(GOBOT_SUPERDEX_ABI_DEFINITIONS
            SuperDex::MochiPhysics INTERFACE_COMPILE_DEFINITIONS)
        if(GOB_REAL_T_IS_DOUBLE)
            set(GOBOT_SUPERDEX_EXPECTED_PRECISION
                "MOCHI_USE_DOUBLE_PRECISION=1")
        else()
            set(GOBOT_SUPERDEX_EXPECTED_PRECISION
                "MOCHI_USE_DOUBLE_PRECISION=0")
        endif()
        if(NOT GOBOT_SUPERDEX_EXPECTED_PRECISION IN_LIST
               GOBOT_SUPERDEX_ABI_DEFINITIONS)
            message(FATAL_ERROR
                "The installed SuperDex SDK precision does not match Gobot RealType; "
                "expected ${GOBOT_SUPERDEX_EXPECTED_PRECISION}.")
        endif()
        get_target_property(GOBOT_SUPERDEX_MOCHI_LIBRARY
            SuperDex::MochiPhysics IMPORTED_LOCATION)
        if(NOT GOBOT_SUPERDEX_MOCHI_LIBRARY)
            message(FATAL_ERROR
                "The installed SuperDex SDK does not expose the Mochi runtime location.")
        endif()
        get_filename_component(GOBOT_SUPERDEX_RUNTIME_DIR
            "${GOBOT_SUPERDEX_MOCHI_LIBRARY}" DIRECTORY)
        set(GOBOT_SUPERDEX_MARL_LIBRARY
            "${GOBOT_SUPERDEX_RUNTIME_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}marl${CMAKE_SHARED_LIBRARY_SUFFIX}.1")
        set(GOBOT_SUPERDEX_SDK_TARGET "")
    else()
        get_filename_component(GOBOT_SUPERDEX_SOURCE_DIR
            "${GOB_SUPERDEX_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${GOBOT_SUPERDEX_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "The SuperDex submodule is missing at '${GOBOT_SUPERDEX_SOURCE_DIR}'. "
                "Initialize it with: git submodule update --init --recursive 3rdparty/project_superdex")
        endif()

        include(ExternalProject)
        set(GOBOT_SUPERDEX_INSTALL_DIR "${CMAKE_BINARY_DIR}/superdex/install")
        set(GOBOT_SUPERDEX_BINARY_DIR "${CMAKE_BINARY_DIR}/superdex/build")
        set(GOBOT_SUPERDEX_MOCHI_LIBRARY
            "${GOBOT_SUPERDEX_INSTALL_DIR}/lib/${GOBOT_SUPERDEX_MOCHI_LIBRARY_NAME}")
        set(GOBOT_SUPERDEX_RUNTIME_DIR
            "${GOBOT_SUPERDEX_INSTALL_DIR}/lib")
        set(GOBOT_SUPERDEX_MARL_LIBRARY
            "${GOBOT_SUPERDEX_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}marl${CMAKE_SHARED_LIBRARY_SUFFIX}.1")
        set(GOBOT_SUPERDEX_DOUBLE_PRECISION OFF)
        if(GOB_REAL_T_IS_DOUBLE)
            set(GOBOT_SUPERDEX_DOUBLE_PRECISION ON)
        endif()

        ExternalProject_Add(gobot_superdex_sdk
            SOURCE_DIR "${GOBOT_SUPERDEX_SOURCE_DIR}"
            BINARY_DIR "${GOBOT_SUPERDEX_BINARY_DIR}"
            INSTALL_DIR "${GOBOT_SUPERDEX_INSTALL_DIR}"
            UPDATE_COMMAND ""
            # Re-enter the SDK's incremental build after a submodule update.
            BUILD_ALWAYS TRUE
            CMAKE_ARGS
                "-DCMAKE_BUILD_TYPE=Release"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                "-DSUPERDEX_PHYSICS_CORE_ONLY=ON"
                "-DMOCHI_BUILD_SHARED=ON"
                "-DMOCHI_USE_DOUBLE_PRECISION=${GOBOT_SUPERDEX_DOUBLE_PRECISION}"
            BUILD_BYPRODUCTS
                "${GOBOT_SUPERDEX_MOCHI_LIBRARY}"
                "${GOBOT_SUPERDEX_MARL_LIBRARY}")

        # Imported include directories must exist when CMake evaluates the target.
        file(MAKE_DIRECTORY "${GOBOT_SUPERDEX_INSTALL_DIR}/include")
        add_library(gobot_superdex_mochi SHARED IMPORTED GLOBAL)
        set_target_properties(gobot_superdex_mochi PROPERTIES
            IMPORTED_LOCATION "${GOBOT_SUPERDEX_MOCHI_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GOBOT_SUPERDEX_INSTALL_DIR}/include"
            INTERFACE_COMPILE_FEATURES cxx_std_20
            INTERFACE_COMPILE_DEFINITIONS
                "MOCHI_PHYSICS_DYNAMICALLY_LINKED=1;MOCHI_USE_DOUBLE_PRECISION=$<BOOL:${GOB_REAL_T_IS_DOUBLE}>"
            INTERFACE_COMPILE_OPTIONS
                "-mbmi2;-mlzcnt;-mpclmul;-mfma;-mf16c;-mavx2")
        add_library(SuperDex::MochiPhysics ALIAS gobot_superdex_mochi)

        add_library(gobot_superdex_physics INTERFACE)
        target_link_libraries(gobot_superdex_physics INTERFACE SuperDex::MochiPhysics)
        add_dependencies(gobot_superdex_physics gobot_superdex_sdk)
        add_library(SuperDex::Physics ALIAS gobot_superdex_physics)
        set(GOBOT_SUPERDEX_SDK_TARGET gobot_superdex_sdk)
    endif()

    if(NOT TARGET SuperDex::Physics)
        message(FATAL_ERROR "The SuperDex SDK did not provide SuperDex::Physics.")
    endif()

    set(GOBOT_SUPERDEX_INSTALL_DIR "${GOBOT_SUPERDEX_INSTALL_DIR}" PARENT_SCOPE)
    set(GOBOT_SUPERDEX_SDK_TARGET "${GOBOT_SUPERDEX_SDK_TARGET}" PARENT_SCOPE)
    set(GOBOT_SUPERDEX_MOCHI_LIBRARY "${GOBOT_SUPERDEX_MOCHI_LIBRARY}" PARENT_SCOPE)
    set(GOBOT_SUPERDEX_RUNTIME_DIR "${GOBOT_SUPERDEX_RUNTIME_DIR}" PARENT_SCOPE)
    set(GOBOT_SUPERDEX_MARL_LIBRARY "${GOBOT_SUPERDEX_MARL_LIBRARY}" PARENT_SCOPE)
endfunction()
