#!/bin/bash
#SBATCH -J one_gpu_INTEL_Img_clas_36000_100_epochs 
#SBATCH -p high 
#SBATCH -N 1 
#SBATCH --ntasks=1 
#SBATCH --constraint=intel 
#SBATCH --gres=gpu:1
#SBATCH --workdir=/homedtic/agilbert 
#SBATCH --mem-per-cpu=5GB 
#SBATCH -o /homedtic/agilbert/NLP_Work/model_output.txt
#SBATCH -e /homedtic/agilbert/NLP_Work/model_error.txt

# load modules you will need for your specific job

module load Python/3.6.4-foss-2017a-test 
module load CUDA/9.0.176
module load PyTorch/1.0.0-foss-2017a-Python-3.6.4-CUDA-9.0.176  # 0.4.0
module load numpy/1.14.0-foss-2017a-Python-3.6.4 
module load sklearn/0.19.1-foss-2017a-Python-3.6.4 

# Run your model or script as in a Linux Terminal

echo "Temporal log directory created."

cd /homedtic/agilbert/NLP_Work/src

echo "All loaded. Running image classification training..."

python Text_Classifier_Neural_Network.py

echo "acabado."
date
