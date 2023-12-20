from setuptools import setup
import os

if os.path.exists(os.path.join(os.getcwd(), ".cache")):
    os.system("cp .cache/xpu/lib/*.so build/lib/tf_backend/")
os.system("cd build/lib && ls | grep -v tf_backend | xargs -I {} rm -rf {}")
setup()
