#!/bin/bash

#SBATCH --account=def-aghodsib
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=0-48:00            # time (DD-HH:MM)
#SBATCH --output=%x-%j.out

module load cuda cudnn python/3.5.2
source pytorch3/bin/activate

python /home/rbbidart/learn-lr/src/evol_cifar_test.py --data_loc /home/rbbidart/project/rbbidart/learn-lr --out_loc /home/rbbidart/project/rbbidart/learn-lr/output --generations 50 --num_schedules 12 --epochs 10
