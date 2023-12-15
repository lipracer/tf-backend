#set(codegen_registry_file ${CMAKE_CURRENT_BINARY_DIR}/ops/native/native_op_registry.cpp)
#set(SOURCES native_ops.cpp codegen_registry_file)

set(SOURCES)
list(APPEND SOURCES native_ops.cpp)

if(ENABLE_REG_CODEGEN)
    list(APPEND SOURCES native_op_registry.cpp)
    set_source_files_properties(native_op_registry.cpp PROPERTIES GENERATED TRUE)
endif()

# file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_be_obj_target(native_ops ${SOURCES})

target_include_directories(native_ops PUBLIC ${CMAKE_CURRENT_LIST_DIR})

if(ENABLE_CUDA_DEVICE)
target_include_directories(native_ops PUBLIC ${DEV_OPS_INCLUDE_DIR})
endif()
