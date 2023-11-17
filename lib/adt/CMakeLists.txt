
set(SOURCES tensor_impl.cpp tensor.cpp allocator.cpp ref_counter_ptr.cpp)

add_library(adt OBJECT ${SOURCES})

target_include_directories(adt PRIVATE ${PROJECT_SOURCE_DIR}/include)
