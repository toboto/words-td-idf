import math
import pandas as pd
from wordstats import save_csv_file


def calculate_idf(df, word, column, total):
    """
    Calculate IDF value of a specific word
    :param df: data frame of analyzing documents
    :type df: pandas.DataFrame
    :param word: the word to calculate
    :param column: column name of word cnt
    :param total: total document number
    :return: IDF value
    :rtype: float
    """
    data = df[df['word'] == word].groupby("word").sum()
    try:
        if column != "all":
            rt = math.log2(total / (data.iloc[0][column] + 1))
        else:
            cnt = data.iloc[0]["cnt_in_head"] + data.iloc[0]["cnt_in_body"]
            rt = math.log2(total / (cnt + 1))
    except Exception as e:
        rt = 0

    return max(0, rt)


def generate_idf():
    source = "output-data/words_in_hair_dryer.csv"
    df0 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])
    source = "output-data/words_in_microwave.csv"
    df1 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])
    source = "output-data/words_in_pacifier.csv"
    df2 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])

    df = df0.append(df1).append(df2)

    print("words", len(df['word'].astype("str").unique()))

    total = len(df['review_id'].unique())
    print("total document", total)

    headers = ["word", "idf_in_head", "idf_in_body", "idf"]
    results = []
    i = 0
    for word in df['word'].astype("str").unique():
        idf_in_head = calculate_idf(df, word, "cnt_in_head", total)
        idf_in_body = calculate_idf(df, word, "cnt_in_body", total)
        idf = calculate_idf(df, word, "all", total)
        results.append([word, idf_in_head, idf_in_body, idf])
        i += 1
        if i % 100 == 0:
            print(i)

    save_csv_file(headers, results, "output-data/idf.csv")


def refresh_tfidf(source, output, idffile):
    idf = pd.read_csv(idffile)
    df = pd.read_csv(source)
    print("Converting ", source, df.shape[0])

    review_df = df.groupby("review_id").sum()
    counter = {'hcnt': 0, 'bcnt': 0, 'cnt': 0}

    def __refresh_row_tfidf_in_head(r):
        widf_frame = idf[idf['word'] == r['word']]
        widf = widf_frame.iloc[0]['idf'] if widf_frame.shape[0] == 1 else 0
        words_cnt_in_head = review_df.loc[r['review_id'], 'cnt_in_head']
        counter['hcnt'] += 1
        if counter['hcnt'] % 100 == 0:
            print(counter['hcnt'])
        return r['cnt_in_head'] / words_cnt_in_head * widf if words_cnt_in_head > 0 else 0

    def __refresh_row_tfidf_in_body(r):
        widf_frame = idf[idf['word'] == r['word']]
        widf = widf_frame.iloc[0]['idf'] if widf_frame.shape[0] == 1 else 0
        words_cnt_in_body = review_df.loc[r['review_id'], 'cnt_in_body']
        counter['bcnt'] += 1
        if counter['bcnt'] % 100 == 0:
            print(counter['bcnt'])
        return r['cnt_in_body'] / words_cnt_in_body * widf if words_cnt_in_body > 0 else 0

    def __refresh_row_tfidf(r):
        widf_frame = idf[idf['word'] == r['word']]
        widf = widf_frame.iloc[0]['idf'] if widf_frame.shape[0] == 1 else 0
        words_cnt = review_df.loc[r['review_id'], 'cnt']
        counter['cnt'] += 1
        if counter['cnt'] % 100 == 0:
            print(counter['cnt'])
        return r['cnt'] / words_cnt * widf if words_cnt > 0 else 0

    df['tfidf_in_head'] = df.apply(__refresh_row_tfidf_in_head, axis=1)
    df['tfidf_in_body'] = df.apply(__refresh_row_tfidf_in_body, axis=1)
    df['tfidf'] = df.apply(__refresh_row_tfidf, axis=1)
    df.to_csv(output)


if __name__ == "__main__":
    generate_idf()

    # Notice: The following calculation may exhaust a long time
    idffile = "output-data/idf.csv"
    source = "output-data/words_in_microwave.csv"
    output = "output-data/words_in_microwave_tfidf.csv"
    refresh_tfidf(source, output, idffile)

    source = "output-data/words_in_hair_dryer.csv"
    output = "output-data/words_in_hair_dryer_tfidf.csv"
    refresh_tfidf(source, output, idffile)

    source = "output-data/words_in_pacifier.csv"
    output = "output-data/words_in_pacifier_tfidf.csv"
    refresh_tfidf(source, output, idffile)

