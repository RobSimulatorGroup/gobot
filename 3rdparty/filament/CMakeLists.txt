
if (NOT CLANG_LIBDIR)
    set(ubuntu_default_llvm_lib_dirs
            /usr/lib/llvm-15/lib
            /usr/lib/llvm-14/lib
            /usr/lib/llvm-13/lib
            /usr/lib/llvm-12/lib
            /usr/lib/llvm-11/lib
            /usr/lib/llvm-10/lib
            /usr/lib/llvm-9/lib
            /usr/lib/llvm-8/lib
            /usr/lib/llvm-7/lib
            )
    foreach(llvm_lib_dir in ${ubuntu_default_llvm_lib_dirs})
        message(STATUS "Searching ${llvm_lib_dir} for libc++ and libc++abi")
        find_library(CPP_LIBRARY    c++    PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
        find_library(CPPABI_LIBRARY c++abi PATHS ${llvm_lib_dir} NO_DEFAULT_PATH)
        if (CPP_LIBRARY AND CPPABI_LIBRARY)
            set(CLANG_LIBDIR ${llvm_lib_dir})
            message(STATUS "CLANG_LIBDIR found in ubuntu-default: ${CLANG_LIBDIR}")
            break()
        endif()
    endforeach()
endif()


find_library(CPP_LIBRARY    c++    PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)
find_library(CPPABI_LIBRARY c++abi PATHS ${CLANG_LIBDIR} REQUIRED NO_DEFAULT_PATH)

message(STATUS "CPP_LIBRARY: ${CPP_LIBRARY}")
message(STATUS "CPPABI_LIBRARY: ${CPPABI_LIBRARY}")

add_library(filament STATIC IMPORTED)
set_target_properties(filament PROPERTIES
        IMPORTED_LOCATION  ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilament.a
        COMPILE_FLAGS "-fno-builtin -stdlib=libc++")
target_link_libraries(filament  INTERFACE
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})


add_library(backend STATIC IMPORTED)
set_target_properties(backend PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbackend.a)
target_link_libraries(backend  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(utils STATIC IMPORTED)
set_target_properties(utils PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libutils.a)
target_link_libraries(utils  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(bluegl STATIC IMPORTED)
set_target_properties(bluegl PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbluegl.a)
target_link_libraries(bluegl  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(ibl-lite STATIC IMPORTED)
set_target_properties(ibl-lite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libibl-lite.a)
target_link_libraries(ibl-lite  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(filaflat STATIC IMPORTED)
set_target_properties(filaflat PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilaflat.a)
target_link_libraries(filaflat  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(filamat STATIC IMPORTED)
set_target_properties(filamat PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilamat.a)
target_link_libraries(filamat  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(geometry STATIC IMPORTED)
set_target_properties(geometry PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libgeometry.a)
target_link_libraries(geometry  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})


add_library(filabridge STATIC IMPORTED)
set_target_properties(filabridge PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilabridge.a)
target_link_libraries(filabridge  INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(bluevk STATIC IMPORTED)
set_target_properties(bluevk PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbluevk.a)
target_link_libraries(bluevk  INTERFACE -lstdc++ -ldl
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})

add_library(vkshaders STATIC IMPORTED)
set_target_properties(vkshaders PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libvkshaders.a)

add_library(smol-v STATIC IMPORTED)
set_target_properties(smol-v PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libsmol-v.a)

add_library(Filament INTERFACE)
target_link_libraries(Filament INTERFACE -lstdc++
        ${CPP_LIBRARY} ${CPPABI_LIBRARY})
target_link_libraries(Filament INTERFACE filament geometry backend utils filaflat filabridge ibl-lite bluevk bluegl smol-v vkshaders)
target_include_directories(Filament INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)



add_executable(matc IMPORTED)
set_target_properties(matc PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/matc)
