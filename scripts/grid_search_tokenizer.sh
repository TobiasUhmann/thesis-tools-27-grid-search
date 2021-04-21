#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_search_tokenizer.py \
> logs/grid_search_tokenizer_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
