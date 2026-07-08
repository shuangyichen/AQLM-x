export CUDA_VISIBLE_DEVICES=0,1,2,3   
export MODEL_PATH=/model-weights/Llama-2-7b-hf/    
export DATASET_PATH=pajama
export SAVE_PATH=/scratch/ssd004/scratch/babaogl4/5x8-Llama7b-OG/


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
#  --load $SAVE_PATH
