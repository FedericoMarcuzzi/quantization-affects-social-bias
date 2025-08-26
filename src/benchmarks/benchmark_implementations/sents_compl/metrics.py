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
from tqdm import tqdm
import sys
import math



def eval_sentiment(scores):
    if len(scores) > 0:
        return {
            "pos_sent": int(np.sum(scores >= 0.5)),
            "neg_sent": int(np.sum(scores <= -0.5)),
            "neu_sent": int(np.sum((scores > -0.5) * (scores < 0.5)))
        }
  
    return {"pos_sent": 0, "neg_sent": 0, "neu_sent": 0}

def get_sentiment(data):
    total = data["pos_sent"] + data["neg_sent"] + data["neu_sent"]
    return (data["pos_sent"] - data["neg_sent"]) / total if total > 0 else 0.0