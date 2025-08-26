# based on: https://github.com/ModelTC/llmc/blob/main/configs/quantization/backend/vllm/smoothquant_w8a8.yml
smoothquant_templ = '''
base:
    seed: &seed 42
model:
    type: {}
    path: {}
    torch_dtype: auto
calib:
    name: pileval
    download: True
    n_samples: 512
    bs: 1
    seq_len: 512
    preproc: txt_general_preproc
    seed: *seed
quant:
    method: SmoothQuant
    weight:
        bit: {}
        symmetric: True
        granularity: per_channel
    act:
        bit: 8
        symmetric: True
        granularity: per_token
save:
    save_vllm: {}
    save_fake: {}
    save_path: {}
'''

# based on: https://github.com/ModelTC/llmc/blob/main/configs/quantization/backend/vllm/awq_w4a16.yml
awq_templ = '''
base:
    seed: &seed 42
model:
    type: {}
    path: {}
    tokenizer_mode: fast
    torch_dtype: auto
calib:
    name: pileval
    download: True
    n_samples: 128
    bs: -1
    seq_len: 512
    preproc: txt_general_preproc
    seed: *seed
quant:
    method: Awq
    weight:
        bit: {}
        symmetric: True
        granularity: per_group
        group_size: 128
        need_pack: True
    special:
        trans: True
        trans_version: v2
        weight_clip: True
    quant_out: True
save:
    save_vllm: {}
    save_fake: {}
    save_path: {}
'''

# based on: https://github.com/ModelTC/llmc/blob/main/configs/quantization/backend/vllm/gptq_w4a16.yml
gptq_templ = '''
base:
    seed: &seed 42
model:
    type: {}
    path: {}
    torch_dtype: auto
calib:
    name: wikitext2
    download: True
    n_samples: 128
    bs: 1
    seq_len: 2048
    preproc: wikitext2_gptq
    seed: *seed
quant:
    method: GPTQ
    weight:
        bit: {}
        symmetric: True
        granularity: per_group
        group_size: 128
        need_pack: True
    special:
        actorder: True
        static_groups: True
        percdamp: 0.01
        blocksize: 128
        true_sequential: True
    quant_out: True
save:
    save_vllm: {}
    save_fake: {}
    save_path: {}
'''

import os
import sys

print("[INFO]: script args: ", " ".join(sys.argv))

strat = sys.argv[1] # "awq", "sq", "gptq"
model_type = sys.argv[2] # "Qwen2"
model_path = sys.argv[3] # "../models/Qwen/Qwen2.5-1.5B"
weight_bit = int(sys.argv[4]) # 3, 4, 8

vllm_save = False
fake_save = False
conf = ""
quant_model_path = ""

if strat == "sq":
    supported_conf = [8] #'Supported quant: w4a16, w8a16, w8a8.'

    if weight_bit in supported_conf:
        vllm_save = True
        fake_str = ""
    else:
        fake_save = True
        fake_str = "_fake"
    
    quant_model_path = model_path + f"_SQ_W{weight_bit}A8" + fake_str
    conf = smoothquant_templ.format(model_type, model_path, weight_bit, vllm_save, fake_save, quant_model_path)

if strat == "gptq":
    supported_conf = [4, 8] #'Supported quant: w4a16, w8a16, w8a8.'

    if weight_bit in supported_conf:
        vllm_save = True
        fake_str = ""
    else:
        fake_save = True
        fake_str = "_fake"
    
    quant_model_path = model_path + f"_GPTQ_W{weight_bit}" + fake_str
    conf = gptq_templ.format(model_type, model_path, weight_bit, vllm_save, fake_save, quant_model_path)

if strat == "awq":
    supported_conf = [4, 8] #'Supported quant: w4a16, w8a16, w8a8.'

    if weight_bit in supported_conf:
        vllm_save = True
        fake_str = ""
    else:
        fake_save = True
        fake_str = "_fake"
    
    quant_model_path = model_path + f"_AWQ_W{weight_bit}" + fake_str
    conf = awq_templ.format(model_type, model_path, weight_bit, vllm_save, fake_save, quant_model_path)

conf_path = "llmc_tool/llmc_config_files/"
os.makedirs(conf_path, exist_ok=True)

model = quant_model_path.split("/")[-1]
with open(conf_path + model + ".yml", "w") as file:
    file.write(conf)

check_model_path = quant_model_path + "/" + ("fake_quant_model" if "fake" in quant_model_path else "vllm_quant_model")
folder_exists = os.path.exists(check_model_path)

if not folder_exists:
    print("[INFO]: quantization started")
    os.system(f"bash llmc_tool/run_llmc.sh {model}.yml")
elif not any(os.scandir(check_model_path)):
    print("[INFO]: empty model folder, removing")
    os.removedirs(check_model_path)
    os.system(f"bash llmc_tool/run_llmc.sh {model}.yml")
else:
    print("[INFO]: quantized model already exists")