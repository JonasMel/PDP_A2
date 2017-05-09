#!/bin/bash -l
#SBATCH -A g2017012
#SBATCH -t 8:00
#SBATCH -p node -n 36

make
module load gcc openmp

mpirun -np 1 ./assign2 2000 1 1 &&

mpirun -np 4 ./assign2 2000 2 2 &&

mpirun -np 9 ./assign2 2000 3 3 &&

mpirun -np 16 ./assign2 2000 4 4 &&

mpirun -np 25 ./assign2 2000 5 5 &&

mpirun -np 36 ./assign2 2000 6 6

