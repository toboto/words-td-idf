import pandas as pd
import numpy as np
import math


def sigmoid(x, a = 1):
    return 1 / (1 + math.exp(-1 * a * x))


from_file = "output-data/weight_microwave_tfidf.csv"
# from_file = "output-data/weight_hair_dryer_tfidf.csv"
from_file = "output-data/weight_pacifier_tfidf.csv"

df = pd.read_csv(from_file)

p = df.aggregate({"top10_idf": [np.std, np.mean]})
idf_std = p['top10_idf']['std']
idf_mean = p['top10_idf']['mean']

df['vine_weight'] = df.apply(
    lambda x: 2 if x['vine'] == 1 else 1, axis=1
)
df['top10_idf_weight'] = df.apply(
    lambda x: 1 + sigmoid(x['top10_idf'] - idf_mean, 1 / idf_std), axis=1
)
df['total_weight'] = df.apply(
    lambda x: x['vine_weight'] * x['purchase_weight'] * x['helpful_weight'] * x['top10_idf_weight'],
    axis=1
)
df.to_csv()
