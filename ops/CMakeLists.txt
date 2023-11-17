set(SOURCES binary.cpp)

add_library(ops OBJECT ${SOURCES})

target_include_directories(ops PRIVATE ${PROJECT_SOURCE_DIR}/include)
