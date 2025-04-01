export CUDA_VISIBLE_DEVICES=0,1   # or e.g. 0,1,2,3
export MODEL_PATH=/model-weights/gemma-2b     # /scratch/ssd004/scratch/chensy/hf_home/models--meta-llama--Llama-2-7b-hf/blobs/2ef41cbc275000b29afe157ba487f0530b8c26dc
export DATASET_PATH=pajama
export SAVE_PATH=/scratch/ssd004/scratch/chensy/AQLM-x/
# export WANDB_PROJECT=MY_AQ_EXPS
# export WANDB_NAME=COOL_EXP_NAME

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=32 \
 --num_codebooks=2 \
 --nbits_per_codebook=8 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.005 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --resume \
 --save $SAVE_PATH