import pandas as pd
import logging
from wordstats import read_stop_words


def getlogger():
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def refresh_word_freq(source, output, stopwords):
    logger = getlogger()

    # df = pd.read_csv("output-data/words_in_microwave_tfidf.csv",
    df = pd.read_csv(source, usecols=["word", "review_id", "cnt_in_head", "cnt_in_body", "cnt",
                                      "star_rating", "review_date",
                                      "helpful_votes", "total_votes", "helpful_rate", "verified_purchase",
                                      "tfidf_in_head", "tfidf_in_body", "tfidf"])
    # df = df.head(300)
    logger.info("documents %d" % df.shape[0])
    reviews_cnt = len(df['review_id'].unique())
    logger.info("reviews_cnt %d" % reviews_cnt)

    word_dict = {}
    word_freq = pd.DataFrame(columns=["word", "freq"])
    idx = 0
    refer = 0
    for name, group in df.groupby(["review_id"]):
        try:
            idx += 1
            if idx % 100 == 0:
                logger.info(idx)
            group = group[group["cnt_in_body"] > 0].sort_values(by="tfidf_in_body", ascending=False).head(50)
            refer += 1 if group.iloc[0]['verified_purchase'] else 0
            for w in group['word'].values:
                if w in stopwords:
                    continue
                if w not in word_dict.keys():
                    word_dict[w] = idx
                    word_freq = word_freq.append([{"word": w, "freq": 1}], ignore_index=True)
                else:
                    word_freq.loc[word_freq["word"] == w, "freq"] += 1
        except Exception as e:
            logger.error(e)


    word_freq = word_freq.sort_values(by="freq", ascending=False, ignore_index=True) #.head(10000)
    word_freq.to_csv(output)


def load_word_freq_division(source):
    df = pd.read_csv(source)
    word_dict = {}
    for idx, row in df.iterrows():
        word_dict[row['word']] = idx
    return word_dict


def load_word_freq(source):
    df = pd.read_csv(source)
    word_dict = {}
    for idx, row in df.iterrows():
        word_dict[row['word']] = row['freq']
    return word_dict


if __name__ == "__main__":
    stopwords = read_stop_words()
    # source = "output-data/words_in_microwave_tfidf.csv"
    # output = "output-data/word_freq_in_microwave.csv"
    # refresh_word_freq(source, output, stopwords)

    source = "output-data/words_in_hair_dryer_tfidf.csv"
    output = "output-data/word_freq_in_hair_dryer.csv"
    refresh_word_freq(source, output, stopwords)

