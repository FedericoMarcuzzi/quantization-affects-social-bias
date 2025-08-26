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


from typing import cast, List
import pandas as pd

from src.benchmarks.base_benchmark import BaseBenchmark, BaseBenchmarkContext
from src.configs.base_benchmark_config import BenchmarkConfig
from src.configs.base_data_config import DataConfig
from src.data.base_data import BaseData
from src.models.base.base_model import BaseModel
from src.models.base.utils import PromptStatistics
from pydantic import Field

from .utils import (
    common_run
)

MAPPER = {
    "gender" : "gender",
    "race" : "race",
    "religion_jc" : "religion",
    "religion_mc" : "religion",
}

class RedditBiasConfig(BenchmarkConfig):
    with_removed_anomalies: bool = False

class RedditBiasDataConfig(DataConfig):
    category: List[str] = Field(default_factory=list, description="List of groups to analyze")

class RedditBiasDataProvider(BaseData):
    def _get_prompt(self, row):
        return row

    def __init__(self, data_context):
        print(f"[INFO] RedditBias: init")

        data_config = data_context.get_data_config()
        self.config = data_config
        self.debug = data_config.debug
        self.subset_size = data_config.subset_size
        self.bias_category = data_config.category
        
        """
            get Reddit Bias dataset from https://github.com/SoumyaBarikeri/RedditBias
        """
        list_df = []
        for group in self.bias_category:
            data = pd.read_csv(f"benchmark_datasets/reddit_bias/{group}.csv", usecols=['comments', 'comments_processed'])
            debug_size = self.subset_size if self.debug else len(data)
            real_subset_size = min(len(data), debug_size)
            data.drop(data.index[real_subset_size:], inplace=True)
            data["bias_type"] = group
            list_df.append(data)
            
        self.data = pd.concat(list_df, ignore_index=True)
        self.data['prep_sent_more'] = self.data['comments'].apply(self._get_prompt)
        self.data['prep_sent_less'] = self.data['comments_processed'].apply(self._get_prompt)

    def get_data(self):
        return self.data

class RedditBias(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = self.context.get_benchmark_config()
        redditbias_config = cast(RedditBiasConfig, self.config)
        self.with_removed_anomalies = redditbias_config.with_removed_anomalies
        self.data_provider = cast(RedditBiasDataProvider, self.context.get_dataset())

    def run(self, model: BaseModel):
        print(f"[INFO] RedditBias: run")

        PromptStatistics.reset()
        statistics = common_run(model, self.data_provider, self.with_removed_anomalies)
        PromptStatistics.dump("RedditBias")
        return statistics