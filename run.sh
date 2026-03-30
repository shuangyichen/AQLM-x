# run the following in terminal:
sbatch \
  --account=ANONYMOUS \
  --nodes=1 \
  --gres=gpu:l40s:4\
  --ntasks-per-node=1 \
  --mem=120G \
  --cpus-per-task=4 \
  --time=40:00:00 \
  quantize.sh
