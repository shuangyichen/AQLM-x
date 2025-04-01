#!/bin/sh
#SBATCH --job-name=eval
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# module load cuda-12.4

# export CUDA_HOME=/pkgs/cuda-12.4
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
. /scratch/ssd004/scratch/chensy/envs/dora_llama
# export CUDA_HOME=/pkgs/cuda-12.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

lsmod | grep -i nvidia
sh quantize.sh