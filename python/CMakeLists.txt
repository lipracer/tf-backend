add_library(tf_backend SHARED tf_backend.cpp)

target_include_directories(tf_backend PRIVATE ${PROJECT_SOURCE_DIR}/include)

foreach(target ${tf_backend_targets})
#target_compile_options(${target} PRIVATE ${backend_opts})
#set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_link_libraries(${tf_backend} PUBLIC $<TARGET_OBJECTS:${target}>)
#message(WARNING "foreach target:${target}")
endforeach()

target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:logger>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:adt>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:op_registry>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:device>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:ops>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:runtime>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:capi>)
target_link_libraries(tf_backend PUBLIC $<TARGET_OBJECTS:native_ops>)


target_link_libraries(tf_backend PRIVATE fmt::fmt-header-only)
target_link_libraries(tf_backend PUBLIC pthread dl)

set_target_properties(tf_backend
    PROPERTIES
    COMPILE_FLAGS ${backend_opts}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/tf_backend
)


if(ENABLE_CUDA_DEVICE)
    target_link_libraries(tf_backend PUBLIC mock_cudart)
    target_link_directories(tf_backend PUBLIC ${CUDA_HOME_DIR}/lib64)
    target_link_directories(tf_backend PUBLIC /usr/local/xcudart/lib)
    target_link_options(tf_backend PRIVATE "-Wl,-rpath,${CUDA_HOME_DIR}/lib64")
    target_link_options(tf_backend PRIVATE "-Wl,-rpath,/usr/local/xcudart/lib")
endif()

if(ENABLE_CUDA_DEVICE)
    target_link_libraries(tf_backend PUBLIC ${DEV_OPS_LIB_NAME})
    target_link_directories(tf_backend PUBLIC ${DEV_OPS_LIB_DIR})
    target_link_options(tf_backend PRIVATE "-Wl,-rpath,${DEV_OPS_LIB_DIR}")
endif()
