#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_vectors.py \
> logs/grid_search_vectors_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
