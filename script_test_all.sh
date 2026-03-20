#!/bin/bash
#SBATCH --job-name=test-all-freezes
#SBATCH --account=engr255s114w26_class
#SBATCH --mail-user=amypang@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=47G
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log

source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate stryker
python test_all_freezes.py
