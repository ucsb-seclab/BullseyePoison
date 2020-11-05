import sys
import time
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
net_repeat = 1 # sys.argv[3]
# eval_poisons_root = sys.argv[4]
poison_num = sys.argv[3]
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 3
print("--net-repeat set to {}".format(net_repeat))
time.sleep(5)

start_id = 0
end_id = 49
i = start_id
while i <= end_id:
    if True:
        cmd = 'bash launch/attack-transfer-18.sh {} {} {} {} {}'.format(gpu, mode, i, net_repeat, poison_num)
        print(cmd)
        subprocess.run(cmd.split())
        
        # cmd = 'bash launch/eval-transfer.sh {} {} {} {}'.format(gpu, i, eval_poisons_root)
        # print(cmd)
        # subprocess.run(cmd.split())
    i += 1


