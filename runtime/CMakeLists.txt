set(SOURCES runtime.cpp)

add_library(runtime OBJECT ${SOURCES})

target_include_directories(runtime PRIVATE ${PROJECT_SOURCE_DIR}/include)
