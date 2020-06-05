import sys
import time
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
net_repeat = sys.argv[3]
eval_poisons_root = sys.argv[4]
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 15
print("--net-repeat set to {}".format(net_repeat))
time.sleep(5)

start_id = 40
end_id = 49
i = start_id
while i <= end_id:
    if i % 16 ==  gpu:
        cmd = 'bash launch/attack-transfer-18.sh {} {} {} {}'.format(gpu, mode, i, net_repeat)
        print(cmd)
        subprocess.run(cmd.split())
        
        cmd = 'bash launch/eval-transfer.sh {} {} {} {}'.format(gpu, i, eval_poisons_root)
        print(cmd)
        subprocess.run(cmd.split())
    i += 1


