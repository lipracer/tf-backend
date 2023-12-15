set(SOURCES op_registry.cpp kernel_context.cpp)

add_be_obj_target(op_registry ${SOURCES})

if(ENABLE_CUDA_DEVICE)
target_include_directories(op_registry PUBLIC ${DEV_OPS_INCLUDE_DIR})
endif()

