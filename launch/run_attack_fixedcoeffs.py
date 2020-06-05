import sys
import time
import subprocess

gpu = int(sys.argv[1])
ty = int(sys.argv[2])
net_repeat = 1

mode = "coeffs_fixed_type_{}".format(ty)
eval_poisons_root = 'chk-black-fixedcoeffs-random/{}/2000'.format(mode)
assert gpu >= 0 and gpu <= 3
assert ty in list(range(1,11))
print("--net-repeat set to {}".format(net_repeat))
time.sleep(5)

start_id = 0
end_id = 50
i = start_id
while i <= end_id:
    if i % 4 ==  gpu:
        cmd = 'bash launch/attack-transfer-18.sh {} {} {} {}'.format(gpu, mode, i, net_repeat)
        print(cmd)
        subprocess.run(cmd.split())
        
        cmd = 'bash launch/eval-transfer.sh {} {} {}'.format(gpu, i, eval_poisons_root)
        print(cmd)
        subprocess.run(cmd.split())
    i += 1
