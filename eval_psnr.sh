#!/bin/bash

#SBATCH --job-name=d-nerv-eval-psnr
#SBATCH --output=outfiles/debug.out.%j
#SBATCH --error=outfiles/debug.out.%j
#SBATCH --time=20:00:00
#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load cuda/11.8.0

source /fs/nexus-scratch/rz0605/mediainrenv/bin/activate

srun bash -c "python ./eval_psnr.py;"
