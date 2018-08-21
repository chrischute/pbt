#!/bin/bash

# Example script to start 16 clients using SLURM

#SBATCH --partition=deep
#SBATCH --time=72:00:00
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# GPUs per node
#SBATCH --gres=gpu:1

#SBATCH --job-name="pbt"
#SBATCH --output=%j_pbt.out

# Send email when the job begins, fails, or ends
#SBATCH --mail-user=chute@stanford.edu
#SBATCH --mail-type=ALL

# Print some useful job information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_PWD "$SLURM_SUBMIT_DIR

# Command to run on each node
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
srun --gres=gpu:1 --exclusive -n1 -N1-1 python train.py --name=inception_PBT --use_pbt=True --pbt_server_url=deep24 &
wait
