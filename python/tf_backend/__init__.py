import os,sys
import tensorflow as tf
from pathlib import Path


module_file_path = os.path.abspath(__file__)

module_directory = os.path.dirname(module_file_path)

module_installation_path = os.path.dirname(os.path.dirname(module_directory))


tf.load_library(os.path.join(module_directory, "libtf_backend.so"))
tf.load_library(os.path.join(module_directory, "tf_plugin.so"))
