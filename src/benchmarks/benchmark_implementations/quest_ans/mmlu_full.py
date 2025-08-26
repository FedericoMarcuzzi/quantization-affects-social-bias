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


from typing import cast

import pandas as pd
from pydantic import BaseModel as PBM

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from pydantic import Field

from datasets import load_from_disk
from src.benchmarks.benchmark_implementations.utils import to_serializable


MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
ALLOWED_SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

class MMLUConfig(BenchmarkConfig):
    max_length: int = Field(None, description="The maximum length of the generated text")

class MMLUDataConfig(DataConfig):
    k_shots: int = Field(5, description="The maximum number of shot exampels")

class MMLUDataProvider(BaseData):
    def _get_few_shot(self, row):
        subject_rows = self.data_dev[self.data_dev["subject"]==row['subject']]
        max_k_shots = min(self.k_shots, len(subject_rows))
        if max_k_shots != self.k_shots:
            print(f"Warning: Not enough examples for {row['subject']}. Only {max_k_shots} examples will be used.")
        return subject_rows[:max_k_shots]

    def _format_few_shot_for_generative(self, row, q_id):
        options = list(MAP.keys())
        return (f"Example {q_id}: "
                + f" \n{row['question']} "
                + f" \n".join([f"{op}: {row['choices'][idx]} " for idx, op in enumerate(options)])
                + f" \nCorrect answer \\boxed{{{options[row['answer']]}}} ")

    def _get_promt_for_generative(self, row):
        few_shots = [self._format_few_shot_for_generative(ex, q_id) for q_id, (_, ex) in enumerate(self._get_few_shot(row).iterrows())]
        choices = row["choices"]
        return (f"You are a helpful multiple-choice question solver. "
                + f" \nBelow are {self.k_shots} example questions and their correct answers. Each question has four possible options (A, B, C, D). "
                + f" \n\n"
                + f" \n\n".join(few_shots) + " "
                + f" \n\nNow, use reasoning to answer the following question. "
                + f" \nPlease think step by step before choosing your answer, and put your final answer as one of the following: \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}. "
                + f" \n\nQuestion: "
                + f" \n{row['question']} "
                + f" \nA: {choices[0]} "
                + f" \nB: {choices[1]} "
                + f" \nC: {choices[2]} "
                + f" \nD: {choices[3]} "
                + f" \n\nPlease reason step by step and then provide your final answer inside a \\boxed{{}}. ")

    def __init__(self, data_context):
        print(f"[INFO] MMLU: init")

        self.config = data_context.get_data_config()
        self.debug = self.config.debug
        self.subset_size = self.config.subset_size
        self.k_shots = self.config.k_shots
        
        """
            get MMLU dataset from https://huggingface.co/datasets/cais/mmlu
        """
        self.data_test = load_from_disk("benchmark_datasets/mmlu/test").to_pandas()
        self.data_dev = load_from_disk("benchmark_datasets/mmlu/dev").to_pandas()

        self.data_test.drop(self.data_test[~self.data_test['subject'].isin(ALLOWED_SUBJECTS)].index, inplace=True)

        debug_size = self.subset_size if self.debug else len(self.data_test)
        real_subset_size = min(len(self.data_test), debug_size)
        self.data_test.drop(self.data_test.index[real_subset_size:], inplace=True)

        self.data_test['prompts'] = self.data_test.apply(self._get_promt_for_generative, axis=1)

    def get_data(self) -> pd.DataFrame:
        return self.data_test

class MMLU(BaseBenchmark):

    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = self.context.get_benchmark_config()
        self.data_provider = cast(MMLUDataProvider, self.context.get_dataset())

    def run(self, model: BaseModel):
        print(f"[INFO] MMLU: run")

        model.reset_statistic()
        data = self.data_provider.get_data()

        PromptStatistics.reset()
        output_gn = model.generate(data["prompts"].to_list(), max_tokens=self.config.max_length)
        PromptStatistics.dump("MMLU")

        results = {
            "aggregated_results" : {"token_statistics" : model.get_statistic()},
            "raw_results" : {
                "generation" : output_gn,
                "ground_truth" : data["answer"].tolist(),
                "subject" : data["subject"].tolist(),
                "question" : data["question"].tolist(),
                "choices" : data["choices"].tolist(),
                "prompts" : data["prompts"].tolist(),
            },
            "benchmark_params" : {"answer_map" : MAP, "max_length" : self.config.max_length},
        }

        return to_serializable(results)