import sys
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
net_repeat = sys.argv[3]
target_num = sys.argv[4]
eval_poisons_root = sys.argv[5]
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 15

ids = [1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19] # we only attack those cars that we already have a good accuracy for the cifar models!
ids = ids[gpu:gpu+1]
for i in ids:
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num)
    print(cmd)
    subprocess.run(cmd.split())
    
    cmd = 'bash launch/eval-end2end.sh {} {} {} {}'.format(gpu, i, eval_poisons_root, target_num)
    print(cmd)
    subprocess.run(cmd.split())
    i += 1
