#!/bin/bash
attackResults=attack-results/100-overlap/end2end-training
for targetnum in 1 5 10
do
python analysis/compare_with_baseline.py --end2end 1 --net-repeats 1 3 --methods mean mean-3 convex --paths $attackResults/mean/2000 $attackResults/mean-3Repeat/2000 $attackResults/convex/1000 --target-num $targetnum --res-path analysis-results/100-overlap/end2end-training
done