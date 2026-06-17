Twitter Sentiment Analysis Project

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
df = pd.read_csv("/content/Twitter_Data.csv.zip")
print(df[['clean_text','category']].head())

print(df.info())
print(df.shape)

"""Sentiment Distribution Analysis"""

print(df['category'].value_counts())

"""Statistics"""

df['text_length'] = df['clean_text'].astype(str).apply(len)

print(df['text_length'].describe())

"""Natural Language Processing"""

df['clean_text'] = df['clean_text'].str.lower()

df['clean_text'] = df['clean_text'].apply(
    lambda x: re.sub(r'[^a-zA-Z ]', '', str(x))
)
print(df['clean_text'].head())

"""Machine Learning Algorithms"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = df.dropna()
X = df['clean_text']
y = df['category']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
   random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

"""Model Accuracy"""

print("Accuracy:", accuracy_score(y_test, y_pred))

"""Feature Engineering"""

from sklearn.feature_extraction.text import TfidfVectorizer
df = df.dropna()
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
print(X.shape)

"""Data Visualization"""

sentiment_count = df['category'].value_counts()
plt.figure(figsize=(6,4))
sentiment_count.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

df['category'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Sentiment Percentage")

plt.show()

df['text_length'] = df['clean_text'].astype(str).fillna('').apply(len)

df['text_length'].hist()

plt.title("Tweet Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")

plt.show()

text = " ".join(df['clean_text'].astype(str))

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud)
plt.axis("off")

plt.show()

"""Top 10 Most Common Words"""

from collections import Counter

words = " ".join(df['clean_text'].astype(str)).split()

common_words = Counter(words).most_common(10)

words = [word[0] for word in common_words]
counts = [word[1] for word in common_words]

plt.bar(words, counts)

plt.title("Top 10 Common Words")
plt.xticks(rotation=45)

plt.show()
