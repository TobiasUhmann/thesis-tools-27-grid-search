#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_optimizer.py \
> logs/grid_search_optimizer_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
