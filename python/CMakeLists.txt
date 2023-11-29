add_library(tf_backend SHARED tf_backend.cpp)

target_include_directories(tf_backend PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:logger>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:adt>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:op_registry>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:device>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:ops>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:runtime>)
target_link_libraries(tf_backend PUBLIC fmt)

set_target_properties(tf_backend
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/tf_backend"
)
