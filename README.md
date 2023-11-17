# tf-backend

- conda env  
`conda create -n tensorflow-1.15 python=3.7`

- install bazelisk  
`go install github.com/bazelbuild/bazelisk@latest`

- update submodule  
`git submodule update --init --recursive`

- config tensorflow  
`python configure.py`

- build and install tensorflow  
`bash script/build_plugin.sh -a`

- build plugin and backend  
`cmake -S . -B build && cmake --build build`

- build whl  
`pip wheel .`

- run test  
    ```
    pip install --force-reinstall tf_backend-1.0.0-py3-none-any.whl
    python test/python/test_init.py
    ```
