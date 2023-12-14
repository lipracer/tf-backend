cmake_minimum_required(VERSION 3.12.0)

project (tf-backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter 
-Wno-unused-function -Wno-unused-result -Wno-missing-field-initializers 
-Wno-unknown-pragmas -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas 
-Wno-strict-overflow -Wno-strict-aliasing -Wno-missing-braces -Wno-uninitialized
-Werror=return-type -Werror)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(Options REQUIRED)

if(ENABLE_REG_CODEGEN)
    find_package(CfgLLVM REQUIRED)
endif()

set(backend_opts -D_GLIBCXX_USE_CXX11_ABI=1)

set(tf_backend_targets)

macro(add_be_obj_target target)
    add_library(${target} OBJECT ${ARGN})
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${target} PRIVATE ${backend_opts})
    target_compile_definitions(${target} PRIVATE FMT_HEADER_ONLY)
    list(APPEND tf_backend_targets ${target})
    if(ENABLE_CUDA_DEVICE)
        target_compile_definitions(${target} PRIVATE USE_CUDA)
    endif()
endmacro(add_be_obj_target)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmtlib/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmtlib/include)

add_subdirectory(compiler)
add_subdirectory(distribution)
add_subdirectory(lib)
add_subdirectory(ops)
add_subdirectory(python)
add_subdirectory(runtime)

if(ENABLE_REG_CODEGEN)
    add_subdirectory(tools)
endif()

# third_party
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmtlib)

if(ENABLE_BUILD_PLUGIN)
add_custom_target(plugin_target COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/script/build_plugin.sh COMMENT "Running custom target command")
else()
add_custom_target(plugin_target COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/script/build_plugin.sh -l COMMENT "Running custom target command")
endif()

add_dependencies(plugin_target tf_backend)

if(ENABLE_REG_CODEGEN)
    add_custom_target(codegen_shell_target
        COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/script/run_codegen_tool.sh ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Running custom target command"
    )
    add_dependencies(codegen_shell_target registry_codegen)
else()
add_custom_target(codegen_shell_target COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/ops/native/op_registry.cpp COMMENT "Running custom target command")
endif()
add_dependencies(ops codegen_shell_target)

add_executable(dummy ${CMAKE_CURRENT_LIST_DIR}/tools/main.cpp)
add_dependencies(dummy plugin_target)

################################
# GTest
################################
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/cc_test/*.cpp")
add_executable(UnitTest ${TEST_SOURCES})

set_target_properties(UnitTest PROPERTIES COMPILE_FLAGS ${backend_opts})

target_compile_options(gtest PRIVATE ${backend_opts})
target_compile_options(gtest_main PRIVATE ${backend_opts})
target_link_libraries(UnitTest gtest gtest_main fmt tf_backend)
link_directories(${CMAKE_BINARY_DIR}/lib/tf_backend)

add_test(NAME UnitTest COMMAND UnitTest)

target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_include_directories(UnitTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
