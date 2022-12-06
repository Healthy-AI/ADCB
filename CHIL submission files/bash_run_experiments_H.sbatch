#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$hhistory" ]
then
    hhistory=3
fi

. load_modules.sh

python run_experiments_H.py -t $hhistory
