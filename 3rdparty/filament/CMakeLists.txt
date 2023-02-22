
add_library(filament STATIC IMPORTED)
set_target_properties(filament PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilament.a)

#add_library(backend STATIC IMPORTED)
#set_target_properties(backend PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbackend.a)
#
#add_library(utils STATIC IMPORTED)
#set_target_properties(utils PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libutils.a)
#
#add_library(bluegl STATIC IMPORTED)
#set_target_properties(bluegl PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbluegl.a)
#
#add_library(ibl-lite STATIC IMPORTED)
#set_target_properties(ibl-lite PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libibl-lite.a)

#add_library(filaflat STATIC IMPORTED)
#set_target_properties(filaflat PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilaflat.a)

#add_library(filamat STATIC IMPORTED)
#set_target_properties(filamat PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilamat.a)

add_library(geometry STATIC IMPORTED)
set_target_properties(geometry PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libgeometry.a)

#add_library(filabridge STATIC IMPORTED)
#set_target_properties(filabridge PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libfilabridge.a)

#add_library(bluevk STATIC IMPORTED)
#set_target_properties(bluevk PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libbluevk.a)
#
#add_library(vkshaders STATIC IMPORTED)
#set_target_properties(vkshaders PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libvkshaders.a)
#
#add_library(smol-v STATIC IMPORTED)
#set_target_properties(smol-v PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64/libsmol-v.a)

add_library(Filament INTERFACE )
target_link_libraries(Filament INTERFACE  filament geometry )
target_include_directories(Filament INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(matc IMPORTED)
set_target_properties(matc PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/matc)
