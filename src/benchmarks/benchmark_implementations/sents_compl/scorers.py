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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from functools import partial
from tqdm import tqdm
import sys
import math
import numpy as np


def get_sentiment_scorer():
    return partial(get_sentiment_score, analyzer=SentimentIntensityAnalyzer())

def get_sentiment_score(completions, analyzer):
    return np.array([analyzer.polarity_scores(completion)["compound"] for completion in tqdm(completions, ncols=120, file=sys.stdout)])

def get_toxicity_scorer(is_cpu_mode, device=0):
    if is_cpu_mode:
        device = "cpu"
    
    toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=device)
    return partial(get_toxicity_bert, toxicity_pipeline=toxicity_pipeline)

def get_toxicity_bert(completions, toxicity_pipeline):
    batch_size = 500
    scores = []
    full_scores = []
    for i in tqdm(range(0, len(completions), batch_size), ncols=120, file=sys.stdout, total=math.ceil(len(completions) / batch_size)):
        batch_results = toxicity_pipeline(completions[i : i + batch_size], max_length=512, truncation=True)
        max_scores = [max(element, key=lambda x: x['score'])["score"] for element in batch_results]
        scores.extend(max_scores)
        full_scores.extend(batch_results)
    return scores, full_scores