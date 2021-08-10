#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=geoid_mcmc
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=etehrlic@ncsu.edu
#SBATCH --partition=high2
#SBATCH --time=6-00:00:00
module load python3/3.8.2
module load openmpi/4.0.5-mpi-io
source /home/rudolph/python3-venv/bin/activate
time python3 aspect_geoid_mcmc.py

