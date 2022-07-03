#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf20007
#SBATCH --output=pf20007.out
#SBATCH --error=pf20007.err
#SBATCH --time=6-23:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5

srun python3 Run_Aso1_0_CPU20007.py  