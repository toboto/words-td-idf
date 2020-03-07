import pandas as pd
import numpy as np
import math


def sigmoid(x, a = 1):
    return 1 / (1 + math.exp(-1 * a * x))


# from_file = "output-data/weight_microwave_tfidf.csv"
# output_file = "output-data/microwave_features_rating.csv"

# from_file = "output-data/weight_hair_dryer_tfidf.csv"
# output_file = "output-data/hair_dryer_features_rating.csv"

from_file = "output-data/weight_pacifier_tfidf.csv"
output_file = "output-data/pacifier_features_rating.csv"

df = pd.read_csv(from_file)

results = pd.DataFrame(columns=["product_id", "features_rating"])
for pid, group in df.groupby(by='product_id'):
    weight = 0
    rating = 0
    for idx, r in group.iterrows():
        weight += r['total_weight']
        rating += r['total_weight'] * r['star_rating']

    rating = rating / weight
    results = results.append([{"product_id": pid, "features_rating": rating}], ignore_index=True)

results.to_csv(output_file, index=False)

