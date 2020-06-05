import sys
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 3

start_id = 0
end_id = 36
i = start_id
while i <= end_id:
    if i % 4 == gpu:
        cmd = 'bash launch/attack-end2end-12.sh {} {} {}'.format(gpu, mode, i)
        print(cmd)
        subprocess.run(cmd.split())
    i += 1
