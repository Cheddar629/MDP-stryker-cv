#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=stryker-cv-training
#SBATCH --account=engr255s114w26_class
#SBATCH --mail-user=yfran@umich.edu
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
conda activate Stryker
python yolo26_final.py