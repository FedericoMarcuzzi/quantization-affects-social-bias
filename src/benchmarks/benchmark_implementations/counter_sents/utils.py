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


def common_run(model, data_provider, with_removed_anomalies=False):
    model.reset_statistic()
    data = data_provider.get_data()

    mr_ppl = model.perplexities(data["prep_sent_more"].to_list())
    ls_ppl = model.perplexities(data["prep_sent_less"].to_list())

    results = {
        "aggregated_results" : {"token_statistics" : model.get_statistic()},
        "raw_results" : {
            "group" : data["bias_type"].tolist(),
            "more_stereo_ppl" : mr_ppl,
            "less_stereo_ppl" : ls_ppl,
            "more_stereo_prompt" : data["prep_sent_more"].tolist(),
            "less_stereo_prompt" : data["prep_sent_less"].tolist(),
        },
        "benchmark_params" : {"with_removed_anomalies" : with_removed_anomalies},
    }

    return results