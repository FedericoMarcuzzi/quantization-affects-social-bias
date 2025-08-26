#!/bin/bash


MODEL_DIR="$1"

bash run.sh DeepSeek-R1-Distill-Llama-8B "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_AWQ_W3 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_AWQ_W3/fake_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_AWQ_W4 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_AWQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_AWQ_W8 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_AWQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_GPTQ_W4 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_GPTQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_GPTQ_W8 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_GPTQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_SQ_W4A8 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_SQ_W4A8/fake_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Llama-8B_SQ_W8A8 "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B_SQ_W8A8/vllm_quant_model" configs/models/default.yaml

bash run.sh DeepSeek-R1-Distill-Qwen-14B "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_AWQ_W3 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_AWQ_W3/fake_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_AWQ_W4 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_AWQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_AWQ_W8 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_AWQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_GPTQ_W4 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_GPTQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_GPTQ_W8 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_GPTQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_SQ_W4A8 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_SQ_W4A8/fake_quant_model" configs/models/default.yaml
bash run.sh DeepSeek-R1-Distill-Qwen-14B_SQ_W8A8 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B_SQ_W8A8/vllm_quant_model" configs/models/default.yaml

bash run.sh Llama-3.1-8B-Instruct "$MODEL_DIR/Llama-3.1-8B-Instruct" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_AWQ_W3 "$MODEL_DIR/Llama-3.1-8B-Instruct_AWQ_W3/fake_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_AWQ_W4 "$MODEL_DIR/Llama-3.1-8B-Instruct_AWQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_AWQ_W8 "$MODEL_DIR/Llama-3.1-8B-Instruct_AWQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_GPTQ_W4 "$MODEL_DIR/Llama-3.1-8B-Instruct_GPTQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_GPTQ_W8 "$MODEL_DIR/Llama-3.1-8B-Instruct_GPTQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_SQ_W4A8 "$MODEL_DIR/Llama-3.1-8B-Instruct_SQ_W4A8/fake_quant_model" configs/models/default.yaml
bash run.sh Llama-3.1-8B-Instruct_SQ_W8A8 "$MODEL_DIR/Llama-3.1-8B-Instruct_SQ_W8A8/vllm_quant_model" configs/models/default.yaml

bash run.sh Qwen2.5-14B-Instruct "$MODEL_DIR/Qwen2.5-14B-Instruct" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_AWQ_W3 "$MODEL_DIR/Qwen2.5-14B-Instruct_AWQ_W3/fake_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_AWQ_W4 "$MODEL_DIR/Qwen2.5-14B-Instruct_AWQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_AWQ_W8 "$MODEL_DIR/Qwen2.5-14B-Instruct_AWQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_GPTQ_W4 "$MODEL_DIR/Qwen2.5-14B-Instruct_GPTQ_W4/vllm_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_GPTQ_W8 "$MODEL_DIR/Qwen2.5-14B-Instruct_GPTQ_W8/vllm_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_SQ_W4A8 "$MODEL_DIR/Qwen2.5-14B-Instruct_SQ_W4A8/fake_quant_model" configs/models/default.yaml
bash run.sh Qwen2.5-14B-Instruct_SQ_W8A8 "$MODEL_DIR/Qwen2.5-14B-Instruct_SQ_W8A8/vllm_quant_model" configs/models/default.yaml