export VLLM_LOGGING_LEVEL=DEBUG
export TRANSFORMERS_VERBOSITY=info

export MODEL=/data/Qwen3.5-9B/
 
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=true
export VLLM_FP32_SOFTMAX=false
export VLLM_GRAPH_RESERVED_MEM=0.2 

export VLLM_SKIP_WARMUP=0
export PT_HPU_LAZY_MODE=0

echo "Running baseline test with prefix caching disabled..."
python test_prefix.py \
  --model $MODEL \
  --mode text \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.5 \
  --prefix-cache-test \
  --shared-prefix-repeats 512


echo "Running prefix caching test with prefix caching enabled..."
python test_prefix.py \
  --model $MODEL \
  --mode text \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.5 \
  --prefix-cache-test \
  --enable-prefix-caching \
  --shared-prefix-repeats 512