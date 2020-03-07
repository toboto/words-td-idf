from wordstats import *
import numpy as np
import matplotlib.pyplot as plt

stopwords = read_stop_words()
source = "raw-data/microwave.csv"
df1 = pd.read_csv(source)
source = "raw-data/hair_dryer.csv"
df2 = pd.read_csv(source)
source = "raw-data/pacifier.csv"
df3 = pd.read_csv(source)
# df = df1.append(df2).append(df3)
df = df3

header = ["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id", "review_date", "customer_id",
          "product_id", "product_parent", "product_title", "product_category", "star_rating",
          "vine", "verified_purchase", "helpful_votes", "total_votes", "helpful_rate"]


df['vine'] = df.apply(lambda x: 1 if x['vine'].lower() == 'y' else 0, axis=1)
print(df[df['vine'] == 1].shape[0])
print(df.shape[0])
print(df[df['vine'] == 1].shape[0] / df.shape[0])

df['helpful_vote_rate'] = df.apply(
    lambda x: x['helpful_votes'] / x['total_votes'] if x['total_votes'] > 0 else 0,
    axis=1)
print(df[df['helpful_vote_rate'] > 0].shape[0])
print(df.shape[0])
print(df[df['helpful_vote_rate'] > 0].shape[0] / df.shape[0])

df['verified_purchase'] = df.apply(lambda x: 1 if x['verified_purchase'].lower() == 'y' else 0, axis=1)
print(df[df['verified_purchase'] == 1].shape[0])
print(df.shape[0])
print(df[df['verified_purchase'] == 1].shape[0] / df.shape[0])


df = pd.read_csv("output-data/words_in_microwave_tfidf.csv", usecols=["tfidf", "star_rating", "review_id"])
df = df.groupby(by="review_id").aggregate({"tfidf": [np.count_nonzero, np.max, np.sum, np.mean, np.std]})
print(df.head(100))


