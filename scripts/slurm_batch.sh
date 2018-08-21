#!/bin/bash

# Example script to start 16 clients using SLURM
# Usage: sbatch scripts/run_clients.sh <PBT_SERVER_URL>

#SBATCH --partition=deep
#SBATCH --time=72:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# GPUs per node
#SBATCH --gres=gpu:4

#SBATCH --job-name="pbt"
#SBATCH --output=%j_pbt.out

# Send email when the job begins, fails, or ends
####SBATCH --mail-user=<YOUR_EMAIL>
####SBATCH --mail-type=END,FAIL

# Print some useful job information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_PWD "$SLURM_SUBMIT_DIR

# Command to run on each node
srun scripts/run_pbt_node.sh 4 $1
