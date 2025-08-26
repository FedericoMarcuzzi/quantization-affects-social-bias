if ! pip show huggingface_hub > /dev/null; then
    pip install huggingface_hub[hf_transfer]
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_DIR=$1

# DeepSeek Qwen
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --exclude "original/*" --local-dir "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B"

# Qwen
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --exclude "original/*" --local-dir "$MODEL_DIR/Qwen2.5-14B-Instruct"

# DeepSeek LLaMA
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --exclude "original/*" --local-dir "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B"

# LLaMA
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --token "$HF_TOKEN" --exclude "original/*" --local-dir "$MODEL_DIR/Llama-3.1-8B-Instruct"
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --token "$HF_TOKEN" --exclude "original/*" --local-dir "$MODEL_DIR/Llama-3.3-70B-Instruct" # Judge