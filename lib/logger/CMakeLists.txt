set(SOURCES 
logger.cpp
)

add_be_obj_target(logger ${SOURCES})

target_include_directories(logger PRIVATE ${PROJECT_SOURCE_DIR}/third_party/fmt/include)
