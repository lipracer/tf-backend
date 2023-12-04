set(SOURCES 
backend_api.cpp
)

add_be_obj_target(capi ${SOURCES})

target_include_directories(logger PRIVATE ${PROJECT_SOURCE_DIR}/third_party/fmt/include)
