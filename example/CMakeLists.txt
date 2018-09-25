file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    add_dependencies(${name} ${LINK_DEPENDS})
    target_link_libraries(${name} tensorflow_cc tensorflow_framework)

endforeach(source)
