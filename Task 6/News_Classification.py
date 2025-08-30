import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# 2. Combine headline + short_description
df['text'] = df['headline'] + " " + df['short_description']

# 3. Clean text
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df['text'] = df['text'].apply(clean_text)

# 4. Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

# 5. Train/Test split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Convert text to TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 8. Predictions
y_pred = model.predict(X_test_tfidf)

# 9. Evaluate
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
