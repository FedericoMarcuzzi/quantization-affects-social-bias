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

class StereoSetConfig(BenchmarkConfig):
    with_removed_anomalies: bool = False

class StereoSetDataConfig(DataConfig):
    category: List[str] = Field(default_factory=list, description="List of groups to analyze")

class StereoSetDataProvider(BaseData):
    def _get_prompt(self, row):
        return row

    def __init__(self, data_context):
        print(f"[INFO] StereoSet: init")

        data_config = data_context.get_data_config()
        self.config = data_config
        self.debug = data_config.debug
        self.subset_size = data_config.subset_size
        self.bias_category = data_config.category

        """
            get StereoSet dataset from https://github.com/moinnadeem/StereoSet
        """
        data = pd.read_json("benchmark_datasets/stereo_set/stereo_set.json")
        intra_data = data["data"]["intrasentence"]
        data_list = []
        for item in intra_data:
            sentences_dict = {s['gold_label']: s['sentence'] for s in item['sentences']}

            data_list.append({
                'bias_type': item['bias_type'],
                'sent_more': sentences_dict.get('stereotype'),
                'sent_less': sentences_dict.get('anti-stereotype'),
            })

        data = pd.DataFrame(data_list)
        data.drop(data[~data["bias_type"].isin(self.bias_category)].index, inplace=True)

        debug_size = self.subset_size if self.debug else len(data)
        real_subset_size = min(len(data), debug_size)
        data.drop(data.index[real_subset_size:], inplace=True)

        data['prep_sent_more'] = data['sent_more'].apply(self._get_prompt)
        data['prep_sent_less'] = data['sent_less'].apply(self._get_prompt)
        self.data = data

    def get_data(self):
        return self.data

class StereoSet(BaseBenchmark):
    def __init__(self, context: BaseBenchmarkContext):
        super().__init__(context)
        self.context = context
        self.config = self.context.get_benchmark_config()
        stereo_set_config = cast(StereoSetConfig, self.config)
        self.with_removed_anomalies = stereo_set_config.with_removed_anomalies
        self.data_provider = cast(StereoSetDataProvider, self.context.get_dataset())

    def run(self, model: BaseModel):
        print(f"[INFO] StereoSet: run")
    
        PromptStatistics.reset()
        statistics = common_run(model, self.data_provider, self.with_removed_anomalies)
        PromptStatistics.dump("StereoSet")
        return statistics