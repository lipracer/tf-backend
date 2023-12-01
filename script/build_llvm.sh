#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_DIR=$SCRIPT_DIR/..

mkdir -p $PROJECT_DIR/llvm_build

LLVM_BUILD_DIR=$PROJECT_DIR/llvm_build

cd $PROJECT_DIR/llvm_build

cmake -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -DCMAKE_BUILD_TYPE=Release -GNinja $PROJECT_DIR/third_party/llvm-project/llvm

cmake --build .