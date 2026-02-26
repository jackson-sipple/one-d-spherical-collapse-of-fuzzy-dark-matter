#!/bin/bash

#SBATCH -J 1k4k3e1
#SBATCH -e test."%x".err
#SBATCH -o test."%x".out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jsipple@sas.upenn.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=highmem

python3.7 run_no_notebook.py