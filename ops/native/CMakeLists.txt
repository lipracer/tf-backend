#set(codegen_registry_file ${CMAKE_CURRENT_BINARY_DIR}/ops/native/native_op_registry.cpp)
#set(SOURCES native_ops.cpp codegen_registry_file)

set(SOURCES native_ops.cpp native_op_registry.cpp)

set_source_files_properties(native_op_registry.cpp PROPERTIES GENERATED TRUE)

# file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_be_obj_target(native_ops ${SOURCES})

target_include_directories(native_ops PUBLIC ${CMAKE_CURRENT_LIST_DIR})
