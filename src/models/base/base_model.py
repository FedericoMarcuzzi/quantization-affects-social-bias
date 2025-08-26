#    Modified by Federico Marcuzzi, INSAIT

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from src.configs.base_model_config import DEVICE, ModelConfig
from src.prompts.prompt_base import BaseChatFormatter
from src.prompts.prompt_chat_formatter import DummyChatFormatter

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, transformers.BatchEncoding]


@dataclass
class ContextContinuations:
    context: str
    continuations: list[str]


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class BaseModel(ABC):
    def __init__(self, config: ModelConfig, *kargs, **kwargs):
        self.config = config

    def get_chat_formatter(self) -> BaseChatFormatter:
        return DummyChatFormatter()

    def get_config(self):
        return self.config

    @abstractmethod
    def loglikelihood(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the log-likelihood of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (log-likelihood, is-exact-match) pairs

        """
        pass

    @abstractmethod
    def perplexities(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """Computes the perplexities of a list of (context, continuation) pairs.

        Args:
            inputs (List[Tuple[str, str]]): List of (context, continuation) pairs

        Returns:
            List[Tuple[float, bool]]: List of (perplexities, is-exact-match) pairs

        """
        pass

    @abstractmethod
    def generate(self, inputs: Union[str, List[str]], **kwargs) -> List[str]:
        """Generates continuations for a list of inputs.

        Args:
            inputs (Union[str, List[str]]): List of inputs
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        pass

    @abstractmethod
    def generate_chat(self, messages: List[List[Message]], **kwargs) -> List[str]:
        """Generates continuations for a list of messages.

        Args:
            messages (List[List[Message]]): List of input messages
            **kwargs: Keyword arguments to pass to the model during generation

        Returns:
            List[str]: List of generated continuations
        """
        pass

    @abstractmethod
    def reset_statistic(self):
        """Resets the model's statistics."""
        pass

    @abstractmethod
    def get_statistic(self) -> Dict[str, int]:
        """Returns the model's statistics.

        Returns:
            Dict[str, int]: Dictionary containing the model's statistics
        """
        return {"num_pmt_toks": 0, "num_gen_toks": 0, "num_tot_prmt": 0, "num_tot_gens": 0}
    
    def get_num_gen_tokens(self, inputs):
        """Returns the number of generated tokens per generation.

        Args:
            inputs: The generations.

        Returns:
            List with the length of each generated continuation.
        """
        pass

    @abstractmethod
    def most_prob_options(self, prompts: List[str], answers: Dict[str, int], get_soft_max: bool = True) -> Dict[str, List[float]]:
        """Returns the most probable options for a list of prompts.

        Args:
            prompts (List[str]): List of input prompts
            answers (List[str]): List of possible answers
            get_soft_max (bool): Whether to return softmax probabilities or not

        Returns:
            Dict[str, List[float]]: Dictionary with answers as keys and their probabilities as values
        """
        pass

class Input:
    """Input class for the model.

    Args:
        input (str): Input string
        max_length (Optional[int]): Maximum length of the output
        until (Optional[List[str]]): List of stop words
        model_args (Optional[Dict[str, Any]]): Model arguments
    """

    def __init__(
        self,
        input: str,
        max_length: Optional[int] = None,
        until: Optional[List[str]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.input = input
        self.until = until
        self.max_length = max_length
        self.model_args = model_args

    @property
    def StrInput(self):
        return self.input

    @StrInput.setter
    def StrInput(self, inp: str):
        self.input = inp