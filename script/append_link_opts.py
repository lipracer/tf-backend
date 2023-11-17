import sys,os

# print(sys.argv)

lines = []
with open(sys.argv[1], 'rt') as f:
    lines = f.readlines()

copts = 'copts = ['
link_opts = 'linkopts = ['
with open(sys.argv[1], 'wt') as f:
    for i,line in enumerate(lines):
        copts_index = line.find(copts)
        link_opts_index = line.find(link_opts)
        if -1 != copts_index:
            f.write(line[0:copts_index + len(copts):] + '"-I' + sys.argv[3] + '",' + line[copts_index + len(copts):])
        elif -1 != link_opts_index:
            opts = sys.argv[2].split(' ')
            opts = ['"' + opt + '"' for opt in opts]
            f.write(line[0:link_opts_index + len(link_opts):] + ','.join(opts) + line[link_opts_index + len(link_opts):])
        else:
            f.write(line)
