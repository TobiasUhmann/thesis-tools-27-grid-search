#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_weight_factor.py \
> logs/grid_search_weight_factor_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
