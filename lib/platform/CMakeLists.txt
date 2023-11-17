set(SOURCES device.cpp)

add_library(device OBJECT ${SOURCES})

target_include_directories(device PRIVATE ${PROJECT_SOURCE_DIR}/include)
