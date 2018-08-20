#!/bin/bash

# Example script to start 16 clients using SLURM

#SBATCH --partition=deep
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="pbt"
#SBATCH --output=pbt-%j.out

# Send email when the job begins, fails, or ends
#SBATCH --mail-user=chute@stanford.edu
#SBATCH --mail-type=ALL

# Print some useful job information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Command to run on each node
srun python scripts/run_client.py --server_url=deep6 --config_path=templates/config.csv
