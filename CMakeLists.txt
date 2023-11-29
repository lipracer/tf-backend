cmake_minimum_required(VERSION 3.12.0)

project (tf-backend)

set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -Wno-unused-parameter 
-Wno-unused-function -Wno-unused-result -Wno-missing-field-initializers 
-Wno-unknown-pragmas -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas 
-Wno-strict-overflow -Wno-strict-aliasing -Wno-missing-braces -Wno-uninitialized -Werror=return-type)

add_subdirectory(compiler)
add_subdirectory(distribution)
add_subdirectory(lib)
add_subdirectory(ops)
add_subdirectory(python)
add_subdirectory(runtime)

# third_party
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmtlib)

add_custom_target(plugin_target
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/script/build_plugin.sh 
    COMMENT "Running custom target command"
)

add_dependencies(plugin_target tf_backend)

add_executable(dummy ${CMAKE_CURRENT_LIST_DIR}/tools/main.cpp)
add_dependencies(dummy plugin_target)

################################
# GTest
################################
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/cc_test/*.cpp")
add_executable(UnitTest ${TEST_SOURCES})
target_link_libraries(UnitTest gtest gtest_main tf_backend fmt)

add_test(NAME UnitTest COMMAND UnitTest)

target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
