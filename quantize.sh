export CUDA_VISIBLE_DEVICES=0,1,2,3   
export MODEL_PATH=/model-weights/Llama-2-7b-hf/    # /scratch/ssd004/scratch/chensy/hf_home/models--meta-llama--Llama-2-7b-hf/blobs/2ef41cbc275000b29afe157ba487f0530b8c26dc
export DATASET_PATH=pajama
export SAVE_PATH=/scratch/ssd004/scratch/babaogl4/5x8-Llama7b-OG/
# export WANDB_PROJECT=MY_AQ_EXPS
# export WANDB_NAME=COOL_EXP_NAME

# liz: increase the number of samples from 1024 to ...
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
#  --load /scratch/ssd004/scratch/babaogl4/5x8-Llama7b-Mat/
