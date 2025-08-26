#    Copyright 2025 Federico Marcuzzi, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from .base_model import BaseModel
from src.configs.base_model_config import DEVICE, ModelProvider

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import numpy as np
import torch


class VLLMLogitsRetriever:
    def __init__(self, tokenizer, options_map):
        keys = list(options_map.keys())
        input_ids = tokenizer(keys, padding=False, truncation=False)['input_ids']
        token_ids = {tok_ids[-1] : options_map[keys[i]] for i, tok_ids in enumerate(input_ids)}
        self.id_to_str = {tok_ids[-1] : keys[i] for i, tok_ids in enumerate(input_ids)}
        self.tok_ids = list(token_ids.keys())
        self.ans_ids = list(token_ids.values())

        self.pred = []
        self.soft_max = {k : [] for k in keys}

    def __call__(self, token_ids, logits):
        if token_ids == ():
            sub_logits = logits[self.tok_ids]
            self.pred.append(self.ans_ids[torch.argmax(sub_logits)])
            soft_max = torch.nn.functional.softmax(sub_logits, dim=0).to(dtype=float).cpu().numpy()

            for id, sm in zip(self.tok_ids, soft_max):
                self.soft_max[self.id_to_str[id]].append(sm)

        return logits
    
    def get_answers(self):
        return self.pred
    
    def get_soft_max(self):
        return self.soft_max

class VLLMCausalLM(BaseModel):
    AUTO_TOKENIZER_CLASS = AutoTokenizer
    AUTO_MODEL_CLASS = LLM

    def __init__(self, config):
        super().__init__(config)

        self._batch_size = config.batch_size
        self._max_gen_toks = config.max_gen_toks
        self._quantized = config.quantized
        self._device = ("cuda" if config.device in [DEVICE.AUTO, DEVICE.CUDA] and torch.cuda.is_available() else config.device.value)
        self._generation_args = config.generation_args
        self._dtype = config.dtype
        self._trust_remote_code = config.trust_remote_code
        self._padding_side = config.padding_side
        self._provider = config.provider
        self._seed = config.seed
        assert (self._provider == ModelProvider.VLLM)

        self.kvc_allowed_quant = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
        self.model_path = config.name
        print(self.model_path)

        self.tokenizer_path = self.model_path if config.tokenizer_name is None else config.tokenizer_name

        if self._device == "cuda":
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 1
        print(f"Using {self.num_gpus} GPUs for VLLM model: {self.model_path}")

        self.model = self._load_model()
        self._max_length = self.model.llm_engine.model_config.max_model_len

        self.tokenizer = self._load_tokenizer()
        self.tokenizer.model_max_length = self._max_length

        self.sampling_params = self._set_sampling_params(config.generation_args)
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0
    
    # TO implement
    def loglikelihood(self, prompts):
        return self._loglikelihood(prompts)[0]
    
    # TO implement
    def perplexities(self, prompts):
        return self._loglikelihood(prompts)[1]

    # TO implement
    def most_prob_options(self, prompts, anwers, get_soft_max=True):
        get_logits = VLLMLogitsRetriever(self.tokenizer, anwers)
        sp = SamplingParams(max_tokens=1, truncate_prompt_tokens=self._max_length, detokenize=False, logits_processors=[get_logits])
        outputs = self.model.generate(prompts, sampling_params=sp)

        for prompts in outputs:
            # statistics:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        if get_soft_max:
            return get_logits.get_soft_max()

        return get_logits.get_answers()

    # TO implement
    def generate(self, input, n=1, max_tokens=None):
        max_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else self._max_gen_toks
        temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : max_tokens, "n" : n})
        outputs = self.model.generate(input, sampling_params=temp_sampling_params)

        list_output = []
        self.num_tot_prmt += len(outputs)
        for prompts in outputs:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_tot_gens += len(prompts.outputs)
            list_gen = []
            for gen in prompts.outputs:
                self.num_gen_toks += len(gen.token_ids)
                list_gen.append(gen.text)
            
            list_output.append(list_gen)

        if len(list_output[0]) == 1:
            list_output = [gen[0] for gen in list_output]

        return list_output
    
    # TO implement
    def generate_chat(self, chats, add_generation_prompt=False, continue_final_message=True, *args, **kwargs):
        tokenized_chat = self.tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
        return self.generate(tokenized_chat, *args, **kwargs)
    
    # TO implement
    def reset_statistic(self):
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0

    # TO implement
    def get_statistic(self):
        return {"num_pmt_toks" : int(self.num_pmt_toks), "num_gen_toks" : int(self.num_gen_toks), "num_tot_prmt" : int(self.num_tot_prmt), "num_tot_gens" : int(self.num_tot_gens)}
    
    # TO implement
    def get_num_gen_tokens(self, texts):
        encodings = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        return [len(ids) for ids in encodings['input_ids']]

    def _load_model(self):
        tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.model_path
        kvargs = {}
        
        if self._quantized in self.kvc_allowed_quant:
            '''
            https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html
            The kv_cache_dtype argument specifies the data type for KV cache storage:
            - "auto": Uses the model’s default “unquantized” data type
            - "fp8" or "fp8_e4m3": Supported on CUDA 11.8+ and ROCm (AMD GPU)
            - "fp8_e5m2": Supported on CUDA 11.8+
            '''

            print(f"Using model with quantized kv_cache_dtype: {self._quantized}")
            kvargs = {"kv_cache_dtype": self._quantized, "calculate_kv_scales" : True}

        return self.AUTO_MODEL_CLASS(model=self.model_path, tokenizer=tokenizer_path, trust_remote_code=self._trust_remote_code, max_num_seqs=self._batch_size, tensor_parallel_size=self.num_gpus, gpu_memory_utilization=0.8, dtype=self._dtype, device=self._device, enforce_eager=True, seed=self._seed, **kvargs)

    def _load_tokenizer(self):
        tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.model_path
        return self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer_path, trust_remote_code=self._trust_remote_code, padding_side=self._padding_side)

    def _set_sampling_params(self, generation_args):
        if len(generation_args) == 0:
            return SamplingParams()
        else:
            return SamplingParams(**generation_args)
    
    def _loglikelihood(self, prompts):
        sp = SamplingParams(prompt_logprobs=0, temperature=1, max_tokens=1, truncate_prompt_tokens=self._max_length)
        with torch.no_grad():
            model_output = self.model.generate(prompts, sampling_params=sp)
        
        log_likelihoods = []
        perplexities = []
        for request in model_output:
            prompt = request.prompt_logprobs[1:]
            list_logprops = [next(iter(logprobs.values())).logprob for logprobs in prompt]
            sum_logprops = np.sum(list_logprops)
            log_likelihoods.append(sum_logprops)
            n_toks = len(list_logprops)
            perplexities.append(np.exp(-sum_logprops/n_toks))

            # statistics:
            self.num_pmt_toks += np.sum(n_toks)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        return log_likelihoods, perplexities

    def _get_temp_updated_sampling_params(self, params):
        new_params = self._generation_args.copy()
        new_params.update(params)
        return SamplingParams(**new_params)