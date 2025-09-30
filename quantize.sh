#!/bin/bash
#SBATCH --job-name=quantize
#SBATCH --output=slurm-%j-MAT35-gemma2b.out
#SBATCH --error=slurm-%j-MAT35-gemma2b.err

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/model-weights/gemma-2b/    
export DATASET_PATH=wikitext2
export SAVE_PATH=/project/aip-khisti/babaogl4/vaughan/5x8-MAT35-gemma2b

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=32 \
 --num_codebooks=5 \
 --nbits_per_codebook=8 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --resume \
 --save $SAVE_PATH

# python main.py $MODEL_PATH $DATASET_PATH \
#  --nsamples=1024 \
#  --val_size=32 \
#  --num_codebooks=5 \
#  --nbits_per_codebook=8 \
#  --in_group_size=8 \
#  --relative_mse_tolerance=0.01 \
#  --finetune_batch_size=32 \
#  --finetune_max_epochs=10 \
#  --finetune_early_stop=3 \
#  --finetune_keep_best \
#  --local_batch_size=1 \
#  --offload_activations \
#  --load /project/aip-khisti/babaogl4/vaughan/5x8-MAT35-gemma2b
