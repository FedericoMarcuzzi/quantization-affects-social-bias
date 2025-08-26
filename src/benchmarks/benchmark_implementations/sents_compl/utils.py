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

import numpy as np


from .metrics import (
    expected_maximum_toxicity,
    toxicity_fraction,
    eval_sentiment,
    get_sentiment
)

def init_dict(data, statistics):
    for group in data["group"].unique():
        statistics["group"][group] = {}
        statistics["sub_group"][group] = {}
        statistics["raw_metrics"][group] = {}

        for sub_group in data[data["group"]==group]["sub_group"].unique():
            statistics["sub_group"][group][sub_group] = {}
            statistics["raw_metrics"][group][sub_group] = {}
    
    return statistics

def get_mean_var(data, columns):
    return {col : [np.nanmean(data[col].to_list()), np.nanvar(data[col].to_list())] for col in columns}

def get_agg_metrics(data, columns, metric):
    res = {}
    for key in columns:
        values = [sub_dict.get(key)[0] if isinstance(sub_dict.get(key), list) else sub_dict.get(key) for sub_dict in data.values() if key in sub_dict]
        if metric == "avg":
            res["avg_" + key] = np.nanmean(values)
        if metric == "sum":
            res[key] = np.nansum(values)
    return res

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def get_metrics(dataset, model, config, eval_toxicity, word_embedder, analyzer, data_name):
    model.reset_statistic()
    data, _ = dataset.get_data()
    
    sent_gen = model.generate_chat(data["final_sent_prompt"].to_list(), max_tokens=config.max_gen_length)
    multiple_gen = model.generate_chat(data["final_toxic_prompt"].to_list(), max_tokens=config.max_gen_length, n=1)

    data["sent_gen"] = sent_gen
    data["multiple_gen"] = multiple_gen

    multiple_gen = np.array(multiple_gen) if isinstance(multiple_gen[0], list) else np.array(multiple_gen).reshape(-1, 1)
    single_gen = [gen[0] for gen in multiple_gen]

    flat_multi_gen = multiple_gen.reshape(-1).tolist()
    flat_toxic_scores, flat_full_scores = eval_toxicity(flat_multi_gen)
    toxic_multi_gen = np.array(flat_toxic_scores).reshape(multiple_gen.shape)
    full_scores_gen = np.array_split(flat_full_scores, len(single_gen))

    get_gender_polarity_score = None
    gp_labels, gp_score = get_gender_polarity_score(single_gen, word_embedder)

    data["toxicity"] = [tox[0] for tox in toxic_multi_gen]
    data["exp_max_toxicity"] = expected_maximum_toxicity(toxic_multi_gen)
    data["toxicity_fraction"] = toxicity_fraction(toxic_multi_gen)
  #  data["nrm_wgt_comp_score"] = get_sentiment_score(sent_gen, analyzer) # normalized_weighted_composite_score
    data["gender_polarity_score"] = gp_score
    data["gender_polarity_labels"] = gp_labels

    statistics = {
        "raw_results" : {},
        "full_scores" : to_serializable(full_scores_gen),
        "token_statistics" : model.get_statistic(),
        "total" : {},
        "group" : {},
        "sub_group" : {},
        "raw_metrics" : {}
    }

    statistics = init_dict(data, statistics)

    glb_metrics = ["toxicity", "exp_max_toxicity", "toxicity_fraction", "nrm_wgt_comp_score", "gender_polarity_score"]
    gini_metrics_1 = ["toxicity", "exp_max_toxicity", "toxicity_fraction"]
    gini_metrics_2 = ["pos_sent", "neg_sent"]
    avg_metrics = ["gini_toxicity", "gini_exp_max_toxicity", "gini_toxicity_fraction", "gini_pos_sent", "gini_neg_sent"]
    get_gender_polarity = None
    eval_gender_polarity = None
    # sub_group statistics
    for group in data["group"].unique():
        for sub_group in data[data["group"]==group]["sub_group"].unique():
            data_sub_group = data[data["sub_group"]==sub_group]

            statistics["raw_metrics"][group][sub_group]["raw_sentiment"] = eval_sentiment(data_sub_group["nrm_wgt_comp_score"])
            statistics["raw_metrics"][group][sub_group]["pos_sent"] = statistics["raw_metrics"][group][sub_group]["raw_sentiment"]["pos_sent"]
            statistics["raw_metrics"][group][sub_group]["neg_sent"] = statistics["raw_metrics"][group][sub_group]["raw_sentiment"]["neg_sent"]
            statistics["raw_metrics"][group][sub_group]["neu_sent"] = statistics["raw_metrics"][group][sub_group]["raw_sentiment"]["neu_sent"]

            statistics["raw_metrics"][group][sub_group]["raw_gender_polarity"] = eval_gender_polarity(data_sub_group)
            statistics["raw_metrics"][group][sub_group]["male_gp"] = statistics["raw_metrics"][group][sub_group]["raw_gender_polarity"]["male_gp"]
            statistics["raw_metrics"][group][sub_group]["female_gp"] = statistics["raw_metrics"][group][sub_group]["raw_gender_polarity"]["female_gp"]
            statistics["raw_metrics"][group][sub_group]["neutral_gp"] = statistics["raw_metrics"][group][sub_group]["raw_gender_polarity"]["neutral_gp"]

            statistics["sub_group"][group][sub_group] = get_mean_var(data_sub_group, glb_metrics)
            statistics["sub_group"][group][sub_group]["sentiment"] = get_sentiment(statistics["raw_metrics"][group][sub_group]["raw_sentiment"])
            statistics["sub_group"][group][sub_group]["gender_polarity"] = get_gender_polarity(statistics["raw_metrics"][group][sub_group]["raw_gender_polarity"])

    # group statistics
    for group in data["group"].unique():
        data_group = data[data["group"]==group]
        statistics["group"][group] = get_mean_var(data_group, glb_metrics)
        statistics["group"][group] |= get_agg_metrics(statistics["sub_group"][group], gini_metrics_1, "gini")
        statistics["group"][group] |= get_agg_metrics(statistics["raw_metrics"][group], gini_metrics_2, "gini")
        statistics["group"][group]["sentiment"] = get_sentiment(get_agg_metrics(statistics["raw_metrics"][group], ["pos_sent", "neg_sent", "neu_sent"], "sum"))
        statistics["group"][group]["gender_polarity"] = get_gender_polarity(get_agg_metrics(statistics["raw_metrics"][group], ["male_gp", "female_gp", "neutral_gp"], "sum"))

    # total statistics
    statistics["total"] |= get_mean_var(data, glb_metrics)
    statistics["total"] |= get_agg_metrics(statistics["group"], avg_metrics, "avg")

    #statistics["raw_metrics"]["ppl_scores"] = ppl_scores

    for c in data.columns: # TODO TO FIX
        statistics["raw_results"][c] = data[c].tolist()

    return to_serializable(statistics)