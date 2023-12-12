set(SOURCES runtime.cpp)

add_be_obj_target(runtime ${SOURCES})

target_include_directories(runtime PUBLIC ${CUDA_HOME_DIR}/include)
