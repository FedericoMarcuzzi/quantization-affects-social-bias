import os
import numpy as np
import re

def extract_predictions(predictions, op_map):
    op_val = list(op_map.keys())
    op_pattern = "|".join(map(re.escape, map(str, op_val)))

    results = []
    for prediction in predictions:
        prep_prediction = "\\boxed{" + prediction + "}"
        matches = re.findall(r"boxed\{(" + op_pattern + r")\}", prep_prediction)
        if len(matches) > 0:
            results.append(op_map[matches[-1]])
        else:
            results.append(-1)

    return results

def pandas_to_dict(df, ext="acc"):
    dict_final = {}
    for k, v in df.to_dict().items():
        new_k = k
        if isinstance(new_k, tuple):
            new_k = '_'.join(new_k)
        new_k += "_" + ext
        dict_final[new_k] = v
    
    return dict_final

def from_pandas_to_dict(df, metrics=["mean", "var"], ext="acc"):
    data = df.to_dict()
    keys = data[metrics[0]].keys()
    dict_keys = {}
    dict_final = {}

    for k in keys:
        new_k = k
        if isinstance(k, tuple):
            new_k = '_'.join(k)
        new_k += "_" + ext

        dict_keys[k] = new_k
        dict_final[new_k] = []

    for m in metrics:
        for k, v in data[m].items():
            dict_final[dict_keys[k]].append(v)
    
    return dict_final

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