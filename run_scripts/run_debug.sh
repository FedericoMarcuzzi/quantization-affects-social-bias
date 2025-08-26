#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_PATH="$1"

cd ..

python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/quest_ans/mmlu_full.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/counter_sents/stereo_set.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/counter_sents/reddit_bias.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/quest_ans/wino_bias.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/quest_ans/discrim_eval.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/quest_ans/discrim_eval_gen.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/quest_ans/dt_fairness.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/sents_compl/bold.yaml --debug_mode --subset_size 100
python3 run.py --model_config configs/models/default.yaml --model "$MODEL_PATH" --batch_size 10 --results_folder="runs_debug_model" configs/sents_compl/dt_toxic.yaml --debug_mode --subset_size 100