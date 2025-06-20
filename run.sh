#!/bin/sh
#SBATCH --job-name=eval
#SBATCH --gres=gpu:a40:4
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH -c 30
#SBATCH --mem=100G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# module load cuda-12.4

# export CUDA_HOME=/pkgs/cuda-12.4
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# . /scratch/ssd004/scratch/chensy/envs/dora_llama # load env
. /scratch/ssd004/scratch/babaogl4/envs/aqlm-x
# export CUDA_HOME=/pkgs/cuda-12.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

lsmod | grep -i nvidia
sh quantize.sh

