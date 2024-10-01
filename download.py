from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(token='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')
model_name = "google/gemma-2b"  # or whatever else you wish to download
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/ssd004/scratch/chensy/model/", torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/scratch/ssd004/scratch/chensy/model/", torch_dtype="auto")

from datasets import load_dataset

ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",cache_dir="/scratch/ssd004/scratch/chensy/dataset/")