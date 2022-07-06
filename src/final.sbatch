#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=final
#SBATCH --output=final.out
#SBATCH --error=final.err
#SBATCH --time=0-01:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G

# Load Python modules
module load python/booth/3.8/3.8.5


srun python3 Final_distbution_20000.py