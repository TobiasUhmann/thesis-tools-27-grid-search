#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_activation.py \
> logs/grid_search_activation_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
