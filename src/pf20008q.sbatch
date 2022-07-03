#!/bin/bash

#SBATCH --job-name=pf20008q
#SBATCH --output=pf20008q.out
#SBATCH --error=pf20008q.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=5
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_CPU20008s.py 
