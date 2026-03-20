#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=stryker-cv-training
#SBATCH --account=engr255s114w26_class
#SBATCH --mail-user=kayleyg@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=47G
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate stryker-vision
time python yolo26_final_kayley_train_13.py