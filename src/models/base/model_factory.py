from src.configs.base_model_config import ModelConfig, ModelProvider

from .base_model import BaseModel
from .dummy_model import DummyModel
from .vllm_model import VLLMCausalLM


def get_model_from_config(config: ModelConfig) -> BaseModel:

    model: BaseModel
    if config.provider == ModelProvider.DUMMY:
        model = DummyModel(config)

    elif config.provider == ModelProvider.VLLM:
        model = VLLMCausalLM(config)

    else:
        raise ValueError("Model {} not supported".format(config))

    return model