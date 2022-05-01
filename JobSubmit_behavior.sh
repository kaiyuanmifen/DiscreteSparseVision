#!/bin/bash
#SBATCH --job-name=imageDis_train
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=65G               # memory (per node)
#SBATCH --time=0-1:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
conda activate GNN
#conda activate GFlownets


data=$1

method=$2

contrastive=$3

seed=$4


python Model_behaviours.py --Data $data --Method $method --contrastive $contrastive --seed $seed 
