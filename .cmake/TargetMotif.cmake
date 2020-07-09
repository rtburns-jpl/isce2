find_package(Motif)

if(MOTIF_FOUND AND TARGET X11::Xt AND NOT TARGET Motif::Motif)
    add_library(Motif::Motif IMPORTED INTERFACE)
    target_include_directories(Motif::Motif
        SYSTEM INTERFACE ${MOTIF_INCLUDE_DIR})
    target_link_libraries(Motif::Motif
        INTERFACE ${MOTIF_LIBRARIES}
        )
    target_link_libraries(Motif::Motif
        INTERFACE X11::Xt
        )
endif()
