#!/bin/bash

#SBATCH --job-name=pf20008s
#SBATCH --output=pf20008s.out
#SBATCH --error=pf20008s.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --cpus-per-task=28
#SBATCH --mem=60G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_CPU20008s.py 
