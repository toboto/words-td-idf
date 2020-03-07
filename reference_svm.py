from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import random
import re
from wordstats import read_stop_words

def split_words_into_list(s, stopwords):
    results = []
    try:
        words = re.split(r'(\w+)', s)
        if len(words) <= 0:
            return results

        for w in words:
            w = re.sub(r"[\(\),.!:&#<>/';\{\}]", '', w)
            w = w.strip().strip("-").strip("_").replace('"', '').lower()
            if len(w) <= 0 or w in stopwords:
                continue
            results.append(w)
    except:
        print("error")

    return results

def preprocess_text(df, section, records, stopwords):
    """

    :param df:
    :type df: pandas.DataFrame
    :param sentences:
    :param category:
    :return:
    """
    for idx, r in df.iterrows():
        try:
            segs = split_words_into_list(r['review_body'], stopwords)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs))   #去左右空格
            segs = list(filter(lambda x:len(x) > 1, segs)) #长度为1的字符
            segs = list(filter(lambda x:x not in stopwords, segs)) #去掉停用词
            records.append((" ".join(segs), r[section]))# 打标签
            if idx % 100 == 0:
                print(idx)
        except Exception as e:
            print(e)
            continue


df = pd.read_csv("raw-data/hair_dryer.csv",
                 usecols=["review_id",
                          "star_rating", "review_date",
                          "helpful_votes", "total_votes", "verified_purchase",
                          "review_headline", "review_body"])
print("documents", df.shape[0])
stopwords = read_stop_words()
records = []
preprocess_text(df, "star_rating", records, stopwords)

random.shuffle(records)
for sentence in records[:10]:
    print(sentence[0], sentence[1])

#用sk-learn对数据切分，分成训练集和测试集
from sklearn.model_selection import train_test_split
x, y = zip(*records)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

#抽取特征，我们对文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    ngram_range=(1,4),  # use ngrams of size 1 and 2
    max_features=20000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)
#用朴素贝叶斯算法进行模型训练
# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
# classifier.fit(vec.transform(x_train), y_train)
# #对结果进行评分
# print(classifier.score(vec.transform(x_test), y_test))

from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print(svm.score(vec.transform(x_test), y_test))
