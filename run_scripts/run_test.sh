#!/bin/bash


export VLLM_WORKER_MULTIPROC_METHOD=spawn

bash run.sh DummyModel models/DummyModel configs/models/dummy_model.yaml