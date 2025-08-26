#!/bin/bash


MODEL_DIR="$1"

python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 4
python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 8
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 4
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 8
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 3
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 4
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Qwen2.5-14B-Instruct" 8

python3 llmc_tool/create_conf.py sq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 4
python3 llmc_tool/create_conf.py sq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 8
python3 llmc_tool/create_conf.py gptq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 4
python3 llmc_tool/create_conf.py gptq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 8
python3 llmc_tool/create_conf.py awq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 3
python3 llmc_tool/create_conf.py awq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 4
python3 llmc_tool/create_conf.py awq Qwen2 "$MODEL_DIR/DeepSeek-R1-Distill-Qwen-14B" 8

python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 4
python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 8
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 4
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 8
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 3
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 4
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/Llama-3.1-8B-Instruct" 8

python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 4
python3 llmc_tool/create_conf.py sq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 8
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 4
python3 llmc_tool/create_conf.py gptq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 8
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 3
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 4
python3 llmc_tool/create_conf.py awq Llama "$MODEL_DIR/DeepSeek-R1-Distill-Llama-8B" 8