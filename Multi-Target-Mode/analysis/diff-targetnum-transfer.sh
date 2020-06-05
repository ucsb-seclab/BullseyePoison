#!/bin/bash
for method in mean mean-3Repeat mean-5Repeat convex
do
  python eval/diff_target_num.py --method $method --path chk-black/$method/2000 --target-nums 1 5 10 --end2end 0
done