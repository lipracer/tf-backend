#########################################################
# config llvm
#########################################################
set(LLVM_BUILD_DIR ${CMAKE_CURRENT_LIST_DIR}/../llvm_build)

list(APPEND CMAKE_MODULE_PATH ${LLVM_BUILD_DIR}/lib/cmake/llvm)
list(APPEND CMAKE_MODULE_PATH ${LLVM_BUILD_DIR}/lib/cmake/clang)

find_package(LLVM 18.0.0 EXACT REQUIRED PATHS ${LLVM_BUILD_DIR})
#find_package(CLANG 18.0.0 EXACT REQUIRED PATHS ${LLVM_BUILD_DIR})

message(WARNING "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${LLVM_BUILD_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${LLVM_BUILD_DIR}/lib)

set(LLVM_BUILD_LIBRARY_DIR ${LLVM_BUILD_DIR}/lib)

include(TableGen)
include(AddLLVM)
include(AddClang)
include(ClangTargets)
include(ClangConfig)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LINK_COMPONENTS support)

add_clang_executable(registry_codegen PARTIAL_SOURCES_INTENDED registry_codegen.cpp)

target_include_directories(registry_codegen PRIVATE
${CMAKE_CURRENT_LIST_DIR}/../third_party/llvm-project/clang/include)

target_include_directories(registry_codegen PRIVATE
${CMAKE_CURRENT_LIST_DIR}/../third_party/llvm-project/llvm/include)

target_include_directories(registry_codegen PRIVATE
${CMAKE_CURRENT_LIST_DIR}/../llvm_build/tools/clang/include)

clang_target_link_libraries(registry_codegen PRIVATE
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  clangToolingRefactoring
  clangdMain
  clangdSupport
  clangFrontend
  clangParse
  clangTidy
)

target_link_libraries(registry_codegen PRIVATE clang)