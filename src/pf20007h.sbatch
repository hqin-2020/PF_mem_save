#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=pf20007h
#SBATCH --output=pf20007h.out
#SBATCH --error=pf20007h.err
#SBATCH --time=1-23:00:00
#SBATCH --partition=highmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G

# Load Python modules
module load python/booth/3.8/3.8.5

scratch_dir="/scratch/${SLURM_JOB_USER}/"
mkdir -p $scratch_dir

srun python3 Run_Aso1_0_CPU20007h.py  