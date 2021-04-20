#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_sampling.py \
> logs/grid_search_sampling_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
