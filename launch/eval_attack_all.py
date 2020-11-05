import sys
import os
import subprocess

gpu = int(sys.argv[1])
poisons_root_path = sys.argv[2]
assert os.path.exists(poisons_root_path)
assert gpu >= 0 and gpu <= 3

max_id = 49
i = 0

while i <= max_id:
    if i % 2 == 2:
        cmd = 'bash launch/eval-transfer.sh {} {} {}'.format(gpu, i, poisons_root_path)
        print(cmd)
        subprocess.run(cmd.split())
    i += 1


