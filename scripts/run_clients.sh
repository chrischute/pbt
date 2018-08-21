#!/bin/bash

# Run a single node with GPUs in PBT
# Usage: run_pbt_node.sh <NUM_GPUS> <PBT_SERVER_URL>

for ((i=0; i<$1; i++)); do
  # TODO: RUN TRAIN SCRIPT HERE
  export CUDA_VISIBLE_DEVICES=$i
  echo $CUDA_VISIBLE_DEVICES
done

wait
