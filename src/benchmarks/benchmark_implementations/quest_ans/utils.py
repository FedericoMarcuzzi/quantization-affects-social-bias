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


BINARY_MAP = {"A" : 0, "B" : 1}

def coreference_resolution_run(model, data, conf_max_length):
    model.reset_statistic()

    output = model.generate(data["prompts"].to_list(), max_tokens=conf_max_length)

    results = {
        "aggregated_results" : {"token_statistics" : model.get_statistic()},
        "raw_results" : {
            "generation" : output,
            "ground_truth" : data["ground_truth"].tolist(),
            "bias_ground_truth" : data["bias_ground_truth"].tolist(),
            "bias" : data["sent_type"].tolist(),
            "group" : data["gender"].tolist(),
            "prompts" : data["prompts"].tolist()
        },
        "benchmark_params" : {"answer_map" : BINARY_MAP, "max_length" : conf_max_length},
    }

    for c in ["orig_data", "clean_sent", "sentence_text", "pronoun", "tokens", "profession", "g", "profession_first_index", "g_first_index", "opt_1", "opt_2"]: # TODO TO FIX
        if c not in results["raw_results"] and c in data.columns:
            results["raw_results"][c] = data[c].tolist()

    return results