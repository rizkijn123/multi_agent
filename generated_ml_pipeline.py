import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.pipeline import Pipeline
import csv

# Load dataset
try:
    df = pd.read_csv(
        "data/data.csv",
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8"
    )
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Check for missing values and drop them
df.dropna(subset=['text', 'label'], inplace=True)

# Preprocessing functions
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def convert_to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

def stemming(text):
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())

# Apply preprocessing
df['text'] = df['text'].apply(remove_special_characters)
df['text'] = df['text'].apply(convert_to_lowercase)
df['text'] = df['text'].apply(remove_stopwords)
df['text'] = df['text'].apply(lemmatization)
df['text'] = df['text'].apply(stemming)

# Feature extraction
X = df['text']
y = df['label']

# Using TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X_vectorized = vectorizer.fit_transform(X)

# Model training and evaluation
skf = StratifiedKFold(n_splits=5)
accuracy_list = []
f1_list = []
precision_list = []
recall_list = []

for train_index, test_index in skf.split(X_vectorized, y):
    X_train, X_test = X_vectorized[train_index], X_vectorized[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy_list.append(accuracy_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred, average='weighted'))
    precision_list.append(precision_score(y_test, y_pred, average='weighted'))
    recall_list.append(recall_score(y_test, y_pred, average='weighted'))

# Print evaluation metrics
print(f"Accuracy: {np.mean(accuracy_list)}")
print(f"F1 Score: {np.mean(f1_list)}")
print(f"Precision: {np.mean(precision_list)}")
print(f"Recall: {np.mean(recall_list)}")

