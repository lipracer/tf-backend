#!/bin/bash

set -e

BINARY_DIR=$1
SOURCE_DIR=$2

echo $BINARY_DIR/tools/registry_codegen -p $BINARY_DIR $SOURCE_DIR/ops/native/native_ops.h --output-path=$BINARY_DIR/ops/native

$BINARY_DIR/tools/registry_codegen -p $BINARY_DIR $SOURCE_DIR/ops/native/native_ops.h --output-path=$BINARY_DIR/ops/native