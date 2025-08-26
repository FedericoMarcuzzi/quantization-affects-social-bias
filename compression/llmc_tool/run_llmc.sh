#!/bin/bash


conf_filename=$1

if [ -z "$conf_filename" ]; then
    echo "The LLMC config file path is missing"
    exit 1
fi

llmc=llmc_tool/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=$conf_filename
config=llmc_tool/llmc_config_files/$conf_filename


nnodes=1
nproc_per_node=1


find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

mkdir -p "llmc_tool/llmc_log/"

nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id \
> llmc_tool/llmc_log/${task_name}.log 2>&1 &

sleep 2
ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > llmc_tool/llmc_log/${task_name}.pid

pid=$(cat llmc_tool/llmc_log/${task_name}.pid)

wait $pid