#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_emb_size.py \
> logs/grid_search_emb_size_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
