#!/bin/bash
for method in mean mean-3Repeat convex
do
  python eval/diff_target_num.py --method $method --path chk-black-end2end/$method/2000 --target-nums 1 5 --end2end 1
done