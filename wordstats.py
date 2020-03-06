import pandas as pd
import re
import csv

def read_stop_words():
    with open("raw-data/stop_words.txt") as f:
        contents = f.read()
        words = contents.split(",")
    return words


def split_words(s, stopwords):
    results = {}
    try:
        words = re.split(r'(\W+)', s)
        if len(words) <= 0:
            return results

        for w in words:
            w = w.strip().strip(",").strip(".").strip("-").strip("_")\
                .strip("(").strip(")").strip("!").strip(":").strip("'").strip('"').lower()
            if len(w) <= 0 or w in stopwords:
                continue

            if w in results.keys():
                results[w] += 1
            else:
                results[w] = 1
    except:
        print("error")

    return results


def save_csv_file(headers, data, filename):
    with open(filename, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(data)
        f.close()


def parse_csv_files(source, output):
    stopwords = read_stop_words()
    df = pd.read_csv(source)

    header = "word,cnt_in_head,cnt_in_body,cnt,review_id,review_date,customer_id"\
        + ",product_id,product_parent,product_title,product_category,star_rating" \
          ",vine,verified_purchase"

    data = []
    for i, row in df.iterrows():
        words_in_head = split_words(str(row['review_headline']), stopwords)
        words_in_body = split_words(str(row['review_body']), stopwords)
        vine = 0 if row['vine'].lower() == 'n' else 1
        verified_purchase = 0 if row['verified_purchase'].lower() == 'n' else 1
        for (w, cnt_in_head) in words_in_head.items():
            if w in words_in_body:
                cnt_in_body = words_in_body[w]
                words_in_body.pop(w)
            else:
                cnt_in_body = 0
            data.append([
                w, cnt_in_head, cnt_in_body, cnt_in_head + cnt_in_body, row['review_id'], row['review_date'], row['customer_id'],
                row['product_id'], row['product_parent'], row['product_title'], row['product_category'], row['star_rating'],
                vine, verified_purchase
                ])

        for (w, cnt_in_body) in words_in_body.items():
            data.append([
                w, 0, cnt_in_body, cnt_in_body, row['review_id'], row['review_date'], row['customer_id'],
                row['product_id'], row['product_parent'], row['product_title'], row['product_category'], row['star_rating'],
                vine, verified_purchase
            ])

    save_csv_file(header, data, output)
    print(output, "finished")


if __name__ == "__main__":
    parse_csv_files("raw-data/hair_dryer.csv", "output-data/words_in_hair_dryer.csv")
    parse_csv_files("raw-data/microwave.csv", "output-data/words_in_microwave.csv")
    parse_csv_files("raw-data/pacifier.csv", "output-data/words_in_pacifier.csv")


