#!/bin/sh

vllm serve ../qwen25-14b \
    --host 0.0.0.0 \
    --port 8000 \
	--trust-remote-code \
    --max-model-len 8196 \
	--max-num-batched-tokens 65536 \
	--max-num-seqs 64 \
	--dtype auto \
	--kv-cache-dtype auto \
	--tensor-parallel-size 1 \
    --enable-lora \
    --lora-modules my-lora=../qwen25-14b-fine-tuned-lora \
	--gpu-memory-utilization 0.85 
