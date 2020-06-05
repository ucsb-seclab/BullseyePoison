#!/bin/bash
attackResults=attack-results/100-overlap/linear-transfer-learning/
python eval/attack_performance_over_ites.py 60 $attackResults/convex/4000 $attackResults/mean/4000 $attackResults/mean-3Repeat/4000 $attackResults/mean-5Repeat/4000
paperplotspath=/Users/9yte/PhD/Research/poison/paper/poison/plots/single-target/transfer
cp chk-black/convex/4000/plots-retrained-for-60epochs/attack-acc-avg.pdf $paperplotspath/attack-acc-avg-CP.pdf
cp chk-black-ourmean/mean/4000/plots-retrained-for-60epochs/attack-acc-avg.pdf $paperplotspath/attack-acc-avg-BP.pdf
cp chk-black-ourmean/mean-3Repeat/4000/plots-retrained-for-60epochs/attack-acc-avg.pdf $paperplotspath/attack-acc-avg-BP3.pdf
cp chk-black-ourmean/mean-5Repeat/4000/plots-retrained-for-60epochs/attack-acc-avg.pdf $paperplotspath/attack-acc-avg-BP5.pdf