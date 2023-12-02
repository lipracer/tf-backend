set(SOURCES native_ops.cpp op_registry.cpp)

set_source_files_properties(op_registry.cpp PROPERTIES GENERATED TRUE)

# file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_be_obj_target(native_ops ${SOURCES})

target_include_directories(native_ops PUBLIC ${CMAKE_CURRENT_LIST_DIR})
