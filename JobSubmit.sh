#!/bin/bash
#SBATCH --job-name=imageDis_train
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=85G               # memory (per node)
#SBATCH --time=0-6:50            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/11.1
conda activate GNN
#conda activate GFlownets



###pretraining
#python Pretrain_MNIST.py --name "MNIST_Suprise"

data=$1

method=$2

contrastive=$3

seed=$4


python Run_training.py --Data $data --Method $method --contrastive ${contrastive} --seed $seed --Epochs 200
