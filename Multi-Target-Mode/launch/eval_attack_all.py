import sys
import os
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
poisons_root_path = sys.argv[3]
assert os.path.exists(poisons_root_path)
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 3

max_id = 50
i = gpu
if i == 0:
    i = 4
while i <= max_id:
    if mode == 'convex':
        poisons_path = '{}/{}/poison_03999.pth'.format(poisons_root_path, i)
    elif mode == 'mean':
        poisons_path = '{}/{}/poison_01000.pth'.format(poisons_root_path, i)
    
    assert os.path.exists(poisons_path)
    cmd = 'bash launch/eval-attack-transfer-18-nodp.sh {} {} {} {}'.format(gpu, mode, i, poisons_path)
    print(cmd)
    subprocess.run(cmd.split())
    i += 4


