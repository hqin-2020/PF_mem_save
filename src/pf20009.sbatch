#!/bin/bash

#SBATCH --job-name=pf20009
#SBATCH --output=pf20009.out
#SBATCH --error=pf20009.err
#SBATCH --time=36:00:00
#SBATCH --partition=bigmem2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=450G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_CPU20009.py 