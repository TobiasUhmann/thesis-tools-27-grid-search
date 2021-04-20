#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_models.py \
> logs/grid_search_models_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
