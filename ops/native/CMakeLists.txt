set(SOURCES native_ops.cpp op_registry.cpp)

# file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_be_obj_target(native_ops ${SOURCES})
