
set(SOURCES 
tensor_impl.cpp 
tensor.cpp 
allocator.cpp 
ref_counter_ptr.cpp 
shape_inference.cpp
)

add_be_obj_target(adt ${SOURCES})
