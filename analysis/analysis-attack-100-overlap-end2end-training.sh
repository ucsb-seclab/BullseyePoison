#!/bin/bash
attackResults=attack-results/100-overlap/end2end-training/
python analysis/attack_performance_over_ites.py 60 $attackResults/convex/1500 $attackResults/mean/1500 $attackResults/mean-3Repeat/1500