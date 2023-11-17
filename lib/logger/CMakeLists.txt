set(SOURCES 
logger.cpp
)

add_library(logger OBJECT ${SOURCES})

target_include_directories(logger PRIVATE ${PROJECT_SOURCE_DIR}/include)
