- 常见问题
  - bazel编译报错  
    1. `bazel clean` 清理下bazel环境，深度清理 `bazel clean --expunge`
    2. `conda activate tf` 切换conda环境
    3. `rm -rf ~/.cache/bazel`


  - ubuntu-20.04编译报错：`gettid`重复声明  
    1. `python script/apply_patch.py`