import joblib
import pandas as pd
import numpy as np
import random
import os

from word_freq import load_word_freq_division
from word_freq import getlogger
import hashlib

logger = getlogger()


def get_vector(freqfile, source, topwords, classify_name):
    m = hashlib.md5()
    m.update(("%s-%s-%s-%s" % (freqfile, source, topwords, classify_name)).encode())
    h = "cache/" + m.hexdigest()
    if os.path.exists(h):
        return joblib.load(h)

    word_dict = load_word_freq_division(freqfile, topwords)
    dim = len(word_dict)
    logger = getlogger()
    logger.info("freq words %d" % dim)

    df = pd.read_csv(source, usecols=["word", "review_id", "cnt_in_head", "cnt_in_body", "cnt",
                                      "star_rating", "review_date",
                                      "helpful_votes", "total_votes", "helpful_rate", "verified_purchase",
                                      "tfidf_in_head", "tfidf_in_body", "tfidf"])


    reviews_cnt = len(df['review_id'].unique())
    logger.info("review_cnt %d" % reviews_cnt)

    x = np.zeros([reviews_cnt, dim + 1])

    i = 0
    for name, group in df.groupby(["review_id"]):
        x[i][dim] = group.iloc[0][classify_name]
        group = group[group["cnt_in_body"] > 0].sort_values(by="tfidf_in_body", ascending=False).head(10)
        for w in group['word'].values:
            if w not in word_dict.keys():
                continue
            d = word_dict[w]
            if d < dim and group[group['word'] == w].shape[0] > 0:
                x[i][d] = group[group['word'] == w].iloc[0]['tfidf_in_body']
        i += 1
        if i % 100 == 0:
            logger.info(i)
    joblib.dump(x, h)
    return x


if __name__ == "__main__":
    classify_name = "star_rating"

    # freqfile = "output-data/word_freq_in_microwave.csv"
    # source = "output-data/words_in_microwave_tfidf.csv"
    # modelfile = "models/microwave_verified_purchase_2017.model"

    freqfile = "output-data/word_freq_in_hair_dryer.csv"
    source = "output-data/words_in_hair_dryer_tfidf.csv"
    modelfile = "models/hair_dryer_verified_purchase_2029.model"

    # freqfile = "output-data/word_freq_in_pacifier.csv"
    # source = "output-data/words_in_pacifier_tfidf.csv"
    # modelfile = "models/pacifier_verified_purchase_2129.model"

    topwords = 10000;
    word_dict = load_word_freq_division(freqfile, topwords)
    dim = len(word_dict)
    x = get_vector(freqfile, source, topwords, classify_name)

    random.shuffle(x)
    y = x[:, dim:]
    y = y.reshape(y.size)
    x = x[:, :dim]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= int(100))

    logger.info("data length %d" % len(y_test))

    clf = joblib.load(modelfile)
    logger.info("score %.3f" % clf.score(x_test, y_test))
    y_predict = clf.predict(x_test)

    for i in range(100):
        print(y_test[i], y_predict[i])

