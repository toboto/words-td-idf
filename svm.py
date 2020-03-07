from sklearn import svm
import joblib
import pandas as pd
import numpy as np
import random
import time

from word_freq import load_word_freq_division
from word_freq import getlogger

logger = getlogger()

classify_name = "star_rating"

# freqfile = "output-data/word_freq_in_microwave.csv"
# source = "output-data/words_in_microwave_tfidf.csv"
# modelfile = "models/microwave_%s_%s.model"

# freqfile = "output-data/word_freq_in_hair_dryer.csv"
# source = "output-data/words_in_hair_dryer_tfidf.csv"
# modelfile = "models/hair_dryer_%s_%s.model"

freqfile = "output-data/word_freq_in_pacifier.csv"
source = "output-data/words_in_pacifier_tfidf.csv"
modelfile = "models/pacifier_%s_%s.model"

word_dict = load_word_freq_division(freqfile, 10000)
dim = len(word_dict)
logger.info("freq words %d" % dim)

# source = "output-data/words_in_microwave_tfidf.csv"
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

random.shuffle(x)
y = x[:, dim:]
y = y.reshape(y.size)
x = x[:, :dim]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= int(0.1 * y.size))
print("train length", len(y_train), "test length", len(y_test))

# clf = svm.SVC(decision_function_shape="ovo", kernel="rbf", gamma='scale')
clf = svm.SVR(kernel="rbf", gamma='scale')
clf.fit(x_train, y_train)
logger.info(clf.score(x_test, y_test))
y_predict = clf.predict(x_test)

e2 = 0
for i in range(len(y_test)):
    e2 += (y_test[i] - y_predict[i]) ** 2
print(e2 / len(y_test))

for i in range(100):
    print(y_test[i], y_predict[i])

joblib.dump(clf, modelfile % (classify_name, time.strftime('%H%M', time.localtime(time.time()))))
