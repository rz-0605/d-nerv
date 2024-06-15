#!/bin/bash

#SBATCH --job-name=d-nerv
#SBATCH --output=outfiles/debug.out.%j
#SBATCH --error=outfiles/debug.out.%j
#SBATCH --time=20:00:00
#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load cuda/11.8.0

source /fs/nexus-scratch/rz0605/mediainrenv/bin/activate

srun bash -c "python train.py --dataset UVG --model_type D-NeRV --model_size XXS --eval_only --weight ./logs/UVG/D-NeRV/Embed1.25_240_256x320_fc_4_5_52_exp2_f8_k3_e300_warm60_b32_lr0.0005_Fusion6_Strd4,2,2,2,2_dist/model_train_best.pth --quant_model --dump_images;"


