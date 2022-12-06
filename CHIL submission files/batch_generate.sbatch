#!/usr/bin/bash

#SBATCH -t 7-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$N" ]
then
    N=10000
fi
if [ -z "$epsilon" ]
then
    epsilon=0.1
fi
if [ -z "$gamma" ]
then
    gamma=2
fi

. load_modules.sh

python generate.py -n $N -g $gamma -e $epsilon
