import os, sys

os.system("sh script/build_plugin.sh -a")

target_line = 43
target_file = ''
target_path = ''

with os.popen("file third_party/tensorflow/bazel-out") as f:
    path = f.readlines()[0].strip().split(' ')[-1]
    target_path = '/'.join(path.split('/')[0:4])


with os.popen(f"find {target_path} -name log_linux.cc") as f:
    target_file = f.readlines()[0].strip()

print(target_file)

lines = []
with open(target_file, 'rt') as f:
    lines = f.readlines()

with open(target_file, 'wt') as f:
    for i, line in enumerate(lines):
        if i == target_line - 1:
            line = '// ' + line
        f.write(line)


