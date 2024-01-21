# Authors: Nunez, Gumba, and Santos (2023)
# Description: This script is used to extract features from the dataset and conduct exploratory data analysis.

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import phonenumbers
import math
from collections import Counter
df = pd.read_csv('smishing_dataset.csv')

# FEATURE 1
df['length'] = df['text'].apply(len)

# FEATURE 2
tokenizer = RegexpTokenizer(r'\w+')
count_vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1), tokenizer=tokenizer.tokenize)
count_vectorizer.fit(df['text'])
count_vectorizer.get_feature_names_out()
df['word_count'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

# FEATURE 3
df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# FEATURE 4
df['upper_count'] = df['text'].apply(lambda x: sum(1 for char in x if char.isupper()))

# FEATURE 5
df['digit_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c.isdigit()]))

# FEATURE 6
df['special_char_count'] = df['text'].apply(lambda x: sum(1 for char in x if char in string.punctuation))

# FEATURE 7
# from https://stackoverflow.com/questions/51919931/regex-to-catch-url
url_re = re.compile(r"""(https?://www\.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+(/[a-zA-Z0-9.@-]+){0,20})|\ (https?://[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+(/[a-zA-Z0-9.@-]+){0,20})|\ (www.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+(/[a-zA-Z0-9.@-]+){0,20})""")

df['has_url'] = df['text'].apply(lambda x: 1 if url_re.search(str(x)) else 0)

# FEATURE 8
email_re = re.compile(r'\S+@\S+')
df['has_email'] = df['text'].apply(lambda x: 1 if email_re.search(str(x)) else 0)

# FEATURE 9
# takes into account misrepresented currency symbols like P1000 instead of ₱1000.
currency_symbol_re = re.compile(r'[$€£¥₹₽₱₴₩₪]|P\d+')

df['has_currency'] = df['text'].apply(lambda x: 1 if currency_symbol_re.search(str(x)) else 0)

# FEATURE 10
def has_phone_number(text):
    region_codes = ["GB", "PH", "ZZ", "US", "CA", "AU", "DE", "ES", "FR", "IT", "JP", "KR", "MX", "RU", "VN"]

    for region_code in region_codes:
        for match in phonenumbers.PhoneNumberMatcher(text, region_code):
            if phonenumbers.is_valid_number(match.number):
                return 1
    return 0

df['has_phone_number'] = df['text'].apply(has_phone_number)

# FEATURE 11
smishing_df = df[df['is_smishing'] == 1]

count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(smishing_df['text'])
feature_names = count_vectorizer.get_feature_names_out()
count_df = pd.DataFrame(data=count_matrix.toarray(), columns=feature_names)

# Display the counts for each word
word_counts = count_df.sum().sort_values(ascending=False)

# track the top 100 words and use it as a feature
top_100_words = word_counts[:100].index.tolist()

df['has_smishing_words'] = df['text'].apply(lambda x: 1 if any(word in x for word in top_100_words) else 0)

# FEATURE 12
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

df['entropy'] = df['text'].apply(lambda x: entropy(x))
