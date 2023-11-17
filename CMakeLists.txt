cmake_minimum_required(VERSION 3.12.0)

project (tf-backend)

set(CMAKE_CXX_FLAGS "-fPIC")

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(compiler)
add_subdirectory(distribution)
add_subdirectory(lib)
add_subdirectory(ops)
add_subdirectory(python)
add_subdirectory(runtime)

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
target_link_libraries(UnitTest gtest gtest_main tf_backend)

add_test(NAME UnitTest COMMAND UnitTest)

target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
