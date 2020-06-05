#!/bin/bash
attackResults=attack-results/100-overlap/end2end-training/
python analysis/compare_with_baseline.py 60 analysis-results/100-overlap/end2end-training $attackResults/convex/1500 $attackResults/mean/1500 $attackResults/mean-3Repeat/1500