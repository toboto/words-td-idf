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


if __name__ == "__main__":
    source = "output-data/words_in_hair_dryer.csv"
    df0 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])
    source = "output-data/words_in_microwave.csv"
    df1 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])
    source = "output-data/words_in_pacifier.csv"
    df2 = pd.read_csv(source, usecols=["word", "cnt_in_head", "cnt_in_body", "cnt", "review_id"])

    df = df0.append(df1).append(df2)

    print(df.shape)
    print("words", len(df['word'].astype("str").unique()))

    print(df.columns)

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

# print(df.head(20))
# print(df['word'].unique())
#
# print("\n\n\n")
#
# print(df.head(20))
# df = df[df['word'] == "and"]
# print(df.groupby("word").sum())
# print(df.head(20))
