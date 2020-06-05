#!/bin/bash
attackResults=attack-results/100-overlap/linear-transfer-learning/
python analysis/compare_with_baseline.py 60 analysis-results/100-overlap/linear-transfer-learning $attackResults/convex/4000 $attackResults/mean/4000 $attackResults/mean-3Repeat/4000 $attackResults/mean-5Repeat/4000