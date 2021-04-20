#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_pooling.py \
> logs/grid_search_pooling_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
