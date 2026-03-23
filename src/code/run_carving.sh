#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=carving_out.log
#SBATCH -e ./logs/seam-carving-err-%J.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

# Compile
gcc -O3 -lm -lnuma --openmp carving.c -o carving

# Run
# srun  carving ../test_images/720x480.png 720x480-out.png --seam_number 80
# srun  carving ../test_images/3840x2160.png 3840x2160-out.png --seam_number 128 --mode dynamic
srun carving ../test_images/3840x2160.png 3840x2160-out.png --seam_number 512 --mode greedy --batch_size 8
#srun  carving valve.png valve-out.png --seam_number 80

# MONITOR SQUEUE every 2 seconds
# for i in {1..5}; do echo -e "\033[36m\n=== [$i/5] $(date '+%Y-%m-%d %H:%M:%S') ===\033[0m"; squeue --me; if [ $i -lt 5 ]; then sleep 2; fi; done