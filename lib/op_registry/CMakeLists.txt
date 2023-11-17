set(SOURCES op_registry.cpp kernel_context.cpp)

add_library(op_registry OBJECT ${SOURCES})

target_include_directories(op_registry PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(op_registry PUBLIC logger)
