#!/usr/bin/bash

#SBATCH -t 7-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$N" ]
then
    N=10000
fi

. load_modules.sh

python run_experiments_N.py -n $N
