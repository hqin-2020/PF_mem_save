#!/bin/bash

#SBATCH --job-name=test2
#SBATCH --output=test2.out
#SBATCH --error=test2.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl*
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_CPU20007.py 