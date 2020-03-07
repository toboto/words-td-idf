import pandas as pd
from word_freq import getlogger


def vine_weight(vine):
    return 2 if vine else 1


def helpful_rate(rate):
    return 1 + rate


source_tfidf = "output-data/words_in_microwave_tfidf.csv"
source_raw = "raw-data/microwave.csv"
output_file = "output-data/weight_microwave_tfidf.csv"

# source_tfidf = "output-data/words_in_hair_dryer_tfidf.csv"
# source_raw = "raw-data/hair_dryer.csv"
# output_file = "output-data/weight_hair_dryer_tfidf.csv"

# source_tfidf = "output-data/words_in_pacifier_tfidf.csv"
# source_raw = "raw-data/pacifier.csv"
# output_file = "output-data/weight_pacifier_tfidf.csv"

df_tfidf = pd.read_csv(source_tfidf, usecols=[
    "tfidf_in_body", "cnt_in_body", "star_rating", "review_id"
])

df = pd.read_csv(source_raw)
df['vine'] = df.apply(lambda x: 1 if x['vine'].lower() == 'y' else 0, axis=1)
df['helpful_vote_rate'] = df.apply(
    lambda x: x['helpful_votes'] / x['total_votes'] if x['total_votes'] > 0 else 0,
    axis=1)
df['verified_purchase'] = df.apply(lambda x: 1 if x['verified_purchase'].lower() == 'y' else 0, axis=1)

purchase_weight = df.shape[0] / df[df['verified_purchase'] == 1].shape[0]

output = pd.DataFrame(columns=[
    "review_id", "review_date", "product_id", "product_category", "star_rating",
    "vine", "vine_weight", "verified_purchase", "purchase_weight",
    "helpful_votes", "total_votes", "helpful_vote_rate", "helpful_weight",
    "total_idf", "top5_idf", "top10_idf"
])


logger = getlogger()
logger.info("total %d" % df.shape[0])
for idx, row in df.iterrows():
    widfs = df_tfidf[df_tfidf['review_id'] == row['review_id']]\
        .sort_values(by="tfidf_in_body", ascending=False)
    total_words = widfs['cnt_in_body'].sum()
    vine_w = vine_weight(row['vine'])
    purchase_w = purchase_weight if row['verified_purchase'] else 1
    help_w = helpful_rate(row['helpful_vote_rate'])
    total_idf = widfs['tfidf_in_body'].sum() * total_words
    top5_idf = widfs.head(5)['tfidf_in_body'].sum() * total_words
    top10_idf = widfs.head(10)['tfidf_in_body'].sum() * total_words
    output = output.append([{
        'review_id': row['review_id'], 'review_date': row['review_date'],
        'product_id': row['product_id'], 'product_category': row['product_category'],
        'star_rating': row['star_rating'],
        'vine': row['vine'], 'vine_weight': vine_w,
        'verified_purchase': row['verified_purchase'], 'purchase_weight': purchase_w,
        'helpful_votes': row['helpful_votes'], 'total_votes': row['total_votes'],
        'helpful_vote_rate': row['helpful_vote_rate'], 'helpful_weight': help_w,
        'total_idf': total_idf, 'top5_idf': top5_idf, 'top10_idf': top10_idf
    }], ignore_index=True)
    if idx % 100 == 0:
        logger.info(idx)

output.to_csv(output_file, index=False)
