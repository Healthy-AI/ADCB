#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$epsilon" ]
then
    epsilon=0.1
fi

. load_modules.sh

python run_experiments_epsilon.py -e $epsilon
