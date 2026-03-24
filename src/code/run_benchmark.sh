#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=seam_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark_out.log
#SBATCH -e ./logs/seam-benchmark-err-%J.log
#SBATCH --hint=nomultithread

set -euo pipefail

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load numactl
mkdir -p logs
gcc -O3 --openmp -DCARVING_NO_MAIN carving.c benchmark.c -lm -lnuma -o benchmark

srun ./benchmark