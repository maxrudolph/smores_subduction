#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=boxslab
# require exclusive use of the nodes (don't share with other users)
#SBATCH --exclusive
# leave this set to high2
#SBATCH --partition=high2
#SBATCH --mail-user=maxrudolph@ucdavis.edu
#SBATCH --mail-type=ALL
# time is in days:HH:MM:SS
#SBATCH --time=0-00:10:00
#SBATCH --switches=1
#
#module unload openmpi
#module load slurm openmpi/4.0.1 intel-mkl/2019.2
#module unload openmpi
#export PATH=/home/rudolph/sw/openmpi-4.0.2/bin:$PATH

# The number following mpirun -n must match (number of nopdes) * (ntasks-per-node) 4*32=128
time mpirun -n 32 \
     ./aspect.fast boxslab_topcrust.prm


