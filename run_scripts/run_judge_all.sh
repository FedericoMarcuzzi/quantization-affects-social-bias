#!/bin/bash


MODELS_LIST=(
"DeepSeek-R1-Distill-Llama-8B"
"DeepSeek-R1-Distill-Llama-8B_AWQ_W3"
"DeepSeek-R1-Distill-Llama-8B_AWQ_W4"
"DeepSeek-R1-Distill-Llama-8B_AWQ_W8"
"DeepSeek-R1-Distill-Llama-8B_GPTQ_W4"
"DeepSeek-R1-Distill-Llama-8B_GPTQ_W8"
"DeepSeek-R1-Distill-Llama-8B_SQ_W4A8"
"DeepSeek-R1-Distill-Llama-8B_SQ_W8A8"

"DeepSeek-R1-Distill-Qwen-14B"
"DeepSeek-R1-Distill-Qwen-14B_AWQ_W3"
"DeepSeek-R1-Distill-Qwen-14B_AWQ_W4"
"DeepSeek-R1-Distill-Qwen-14B_AWQ_W8"
"DeepSeek-R1-Distill-Qwen-14B_GPTQ_W4"
"DeepSeek-R1-Distill-Qwen-14B_GPTQ_W8"
"DeepSeek-R1-Distill-Qwen-14B_SQ_W4A8"
"DeepSeek-R1-Distill-Qwen-14B_SQ_W8A8"

"Llama-3.1-8B-Instruct"
"Llama-3.1-8B-Instruct_AWQ_W3"
"Llama-3.1-8B-Instruct_AWQ_W4"
"Llama-3.1-8B-Instruct_AWQ_W8"
"Llama-3.1-8B-Instruct_GPTQ_W4"
"Llama-3.1-8B-Instruct_GPTQ_W8"
"Llama-3.1-8B-Instruct_SQ_W4A8"
"Llama-3.1-8B-Instruct_SQ_W8A8"

"Qwen2.5-14B-Instruct"
"Qwen2.5-14B-Instruct_AWQ_W3"
"Qwen2.5-14B-Instruct_AWQ_W4"
"Qwen2.5-14B-Instruct_AWQ_W8"
"Qwen2.5-14B-Instruct_GPTQ_W4"
"Qwen2.5-14B-Instruct_GPTQ_W8"
"Qwen2.5-14B-Instruct_SQ_W4A8"
"Qwen2.5-14B-Instruct_SQ_W8A8"
)

RES_PATH="results/runs"
JUDGE_PATH="models/Llama-3.3-70B-Instruct"

echo "Judge model: $JUDGE_PATH"

for MODEL_NAME in "${MODELS_LIST[@]}"; do
    MODEL_RES_PATH="$RES_PATH/$MODEL_NAME"
   
    LATEST_FOLDER=$(ls -1t "../$MODEL_RES_PATH" | head -n 1)
    LATEST_FOLDER_PATH="$MODEL_RES_PATH/$LATEST_FOLDER"

    echo "Processing model: $MODEL_NAME"
    echo "Latest folder: $LATEST_FOLDER_PATH"

    ./run_judge.sh $LATEST_FOLDER_PATH bold $MODEL_NAME $JUDGE_PATH
    ./run_judge.sh $LATEST_FOLDER_PATH dt_toxic $MODEL_NAME $JUDGE_PATH
done

