#!/bin/bash

#SBATCH --job-name=pf20007q
#SBATCH --output=pf20007q.out
#SBATCH --error=pf20007q.err
#SBATCH --time=36:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=60G

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_CPU20007s.py 
