#!/bin/sh
attackResults=attack-results/100-overlap/linear-transfer-learning/
for targetnum in 1 5 10
do
  python analysis/compare_with_baseline.py --end2end 0 --net-repeats 1 3 5 --methods mean mean-3 mean-5 convex --paths $attackResults/mean/2000 $attackResults/mean-3Repeat/2000 $attackResults/mean-5Repeat/2000 $attackResults/convex/2000 --target-num $targetnum --res-path analysis-results/100-overlap/linear-transfer-learning
done