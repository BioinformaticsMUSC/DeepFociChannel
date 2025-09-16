#!/bin/bash
# Author: Andrew Lutsky
# Last Modified: Sep 16 2025

#SBATCH --job-name  DeepFociDownload 
#SBATCH --time 8:00:00
#SBATCH --nodes 1
#SBATCH --mem 16gb
#SBATCH --partition musc3
#SBATCH --cpus-per-task 1

pwd
python3 retrieve_data.py 
