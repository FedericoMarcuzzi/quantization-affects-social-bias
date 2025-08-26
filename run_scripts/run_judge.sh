#!/bin/bash

# Example usage: bash run_judge.sh results/runs/Qwen2.5-14B-Instruct/1984-04-30_00:00:00 bold Qwen2.5-14B-Instruct models/Llama-3.3-70B-Instruct


export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd ..

RES_PATH="$1"
BENCH_NAME="$2"
MODEL_NAME="$3"
JUDGE_PATH="${4:-}"

BECH_PATH="${RES_PATH}/${BENCH_NAME}"

CMD=(python3 helper_tools/eval_gen_with_judge.py --gen_path "$BECH_PATH")
[ -n "$JUDGE_PATH" ] && CMD+=(--judge_path "$JUDGE_PATH")
"${CMD[@]}"

python3 helper_tools/results_processor.py --parent_dir=$RES_PATH --model_name=$MODEL_NAME --update_res true