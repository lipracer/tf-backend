from setuptools import setup
import os

os.system("cd build/lib && ls | grep -v tf_backend | xargs -I {} rm -rf {}")
setup()
