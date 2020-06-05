#!/bin/bash
attackResults=attack-results/50-overlap/linear-transfer-learning/
python analysis/compare_with_baseline.py 60 analysis-results/50-overlap/linear-transfer-learning $attackResults/convex/1500 $attackResults/mean/1500 $attackResults/mean-3Repeat/1500 $attackResults/mean-5Repeat/1500