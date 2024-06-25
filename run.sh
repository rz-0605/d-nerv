#!/bin/bash

#SBATCH --job-name=d-nerv
#SBATCH --output=outfiles/debug.out.%j
#SBATCH --error=outfiles/debug.out.%j
#SBATCH --time=30:00:00
#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.8.0

source /fs/nexus-scratch/rz0605/mediainrenv/bin/activate

srun bash -c "python train.py --dataset UVG --model_type D-NeRV --model_size XXS -e 300 -b 1 --lr 5e-4 --loss_type Fusion6;"
