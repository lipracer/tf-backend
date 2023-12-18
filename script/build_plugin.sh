#!/bin/bash

set -e

TF_PLUGIN_NAME=tf1.x_plugin

TENSORFLOW_LIB_NAME='libtensorflow_framework.so.1'
TENSORFLOW_LIB_BAZEL_DIR='bazel-out/k8-py2-opt/bin/tensorflow'

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_DIR=$SCRIPT_DIR/..
PROJECT_BUILD_DIR=$PROJECT_DIR/build
TENSORFLOW_DIR=$PROJECT_DIR/third_party/tensorflow

TENSORFLOW_LIB_DIR=$PROJECT_DIR/third_party/tensorflow/$TENSORFLOW_LIB_BAZEL_DIR

PLUGIN_DIR=$TENSORFLOW_DIR/tensorflow/compiler/jit

cp -r $PROJECT_DIR/$TF_PLUGIN_NAME $PLUGIN_DIR
cp -r $PROJECT_DIR/include $PLUGIN_DIR/$TF_PLUGIN_NAME

# BAZEL_OUT_DIR=`readlink -f $TENSORFLOW_DIR/bazel-out`

PYTHON_PAK_DIR=`pip show tensorflow | grep Location | awk '{print $2}'`
TF_LIB_DIR=$PYTHON_PAK_DIR/tensorflow_core

LINK_OPT_TF_FRAMEWORK="-L$TF_LIB_DIR -Wl,--as-needed -l:$TENSORFLOW_LIB_NAME -Wl,-rpath,$TF_LIB_DIR -L$PROJECT_BUILD_DIR/lib/tf_backend -Wl,--as-needed -l:libtf_backend.so -Wl,-rpath,$PYTHON_PAK_DIR/tf_backend -L$TF_LIB_DIR/python -Wl,--as-needed -l:_pywrap_tensorflow_internal.so -Wl,-rpath,$TF_LIB_DIR/python"
# LINK_OPT_TF_FRAMEWORK="-L$TF_LIB_DIR -l:$TENSORFLOW_LIB_NAME -L$PROJECT_BUILD_DIR/lib/tf_backend -l:libtf_backend.so -L$TF_LIB_DIR/python -l:_pywrap_tensorflow_internal.so"

python $SCRIPT_DIR/append_link_opts.py $PLUGIN_DIR/$TF_PLUGIN_NAME/BUILD "$LINK_OPT_TF_FRAMEWORK" $PROJECT_DIR/include
cd $TENSORFLOW_DIR
echo '0.26.1' > .bazelversion
# bazelisk clean

build_tensorflow() {
bazelisk build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install --force-reinstall /tmp/tensorflow_pkg/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
pip install protobuf==3.19.0
}

if [ "-a" = "$1" ]; then
    build_tensorflow
    exit 0
fi

BAZEL_DEBUG_OPT='--compilation_mode=dbg --copt=-g'

# build debug default
# bazelisk build $BAZEL_DEBUG_OPT //tensorflow/compiler/jit/$TF_PLUGIN_NAME:tf_plugin.so
# cp $PROJECT_DIR/third_party/tensorflow/bazel-out/k8-dbg/bin/tensorflow/compiler/jit/tf1.x_plugin/tf_plugin.so $PROJECT_DIR/build/lib/tf_backend

# build release default

if [ ! "-l" = "$1" ]; then
bazelisk build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/compiler/jit/$TF_PLUGIN_NAME:tf_plugin.so
cp $PROJECT_DIR/third_party/tensorflow/bazel-out/k8-opt/bin/tensorflow/compiler/jit/tf1.x_plugin/tf_plugin.so $PROJECT_DIR/build/lib/tf_backend
rm -rf $TENSORFLOW_DIR/tensorflow/compiler/jit/$TF_PLUGIN_NAME
rm -f .bazelversion
# cp $PROJECT_DIR/build/lib/tf_backend/tf_plugin.so $PROJECT_DIR/resource/
else
cp $PROJECT_DIR/resource/tf_plugin.so $PROJECT_DIR/build/lib/tf_backend
fi

cd -
