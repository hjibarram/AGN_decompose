#!/bin/bash
#SBATCH --job-name=NAMEJOB
#SBATCH --output=OUTPUTPATH/values_job.out
#SBATCH --partition=LocalQ
#
#SBATCH --time=900:00:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source PATHJOB/job_file > PATHJOB/job_file.o 2> PATHJOB/job_file.e
wait
echo "DONE"
