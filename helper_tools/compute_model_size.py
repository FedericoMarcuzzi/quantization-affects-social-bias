import json
from pathlib import Path

MODELS_ROOT_PATH = "models/"
MODELS_PATH = [
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B_AWQ_W4",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B_AWQ_W8",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B_GPTQ_W4",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B_GPTQ_W8",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Llama-8B_SQ_W8A8",

    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B_AWQ_W4",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B_AWQ_W8",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B_GPTQ_W4",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B_GPTQ_W8",
    f"{MODELS_ROOT_PATH}/DeepSeek-R1-Distill-Qwen-14B_SQ_W8A8",

    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct",
    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct_AWQ_W4",
    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct_AWQ_W8",
    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct_GPTQ_W4",
    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct_GPTQ_W8",
    f"{MODELS_ROOT_PATH}/Llama-3.1-8B-Instruct_SQ_W8A8",

    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct",
    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct_AWQ_W4",
    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct_AWQ_W8",
    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct_GPTQ_W4",
    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct_GPTQ_W8",
    f"{MODELS_ROOT_PATH}/Qwen2.5-14B-Instruct_SQ_W8A8",
]

def get_model_size(dir_path: Path) -> int:
    index_file = dir_path / "model.safetensors.index.json"
    if index_file.is_file():  
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
                return index.get("metadata", {}).get("total_size", 0)
        except Exception:
            pass
    return sum(f.stat().st_size for f in dir_path.glob("*.safetensors") if f.is_file())

if __name__ == "__main__":
    print(f"{'MODEL NAME':<45} {'SIZE (GB)':>10}")
    print("-" * 60)
    l = []
    for model_path in MODELS_PATH:
        model_name = model_path.split("/")[-1]
        model_path = Path(model_path)
        if (model_path / "vllm_quant_model").exists():
            model_path = model_path / "vllm_quant_model"

        size_bytes = get_model_size(model_path)

        dict_results = {}
        dict_results["model_name"] = model_name
        dict_results["model_path"] = str(model_path)
        dict_results["size_str"] = f"{size_bytes / 1024 ** 3:10.2f}".strip()
        dict_results["size_bytes"] = size_bytes
        l.append(dict_results)

        print(f"{str(model_path):<45} {size_bytes / 1024 ** 3:10.2f}")

        output_file = Path("../results/models_size.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(l, f, indent=4)