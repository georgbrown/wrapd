add_executable( quasi ${CMAKE_CURRENT_SOURCE_DIR}/quasi.cpp )
target_link_libraries( quasi wrapd ${LINKED_LIBS} )