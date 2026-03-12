import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score

# ✅ Step 1: Load dataset
df = pd.read_csv("news_dataset_10000.csv")

# ✅ Step 2: Clean labels
df['label'] = df['label'].str.lower().str.strip()

# ✅ Step 3: Drop missing values
df = df.dropna(subset=['text', 'label'])

# ✅ Step 4: Balance dataset
df_real = df[df['label'] == 'real']
df_fake = df[df['label'] == 'fake']

if len(df_real) > len(df_fake):
    df_fake = resample(df_fake, replace=True, n_samples=len(df_real), random_state=42)
else:
    df_real = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)

df_balanced = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42)

print(f"✅ Balanced dataset: {len(df_balanced)} samples ({df_balanced['label'].value_counts().to_dict()})")

# ✅ Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=42)

# ✅ Step 6: Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Step 7: Train model
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# ✅ Step 8: Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n📊 Model Performance:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ✅ Step 9: Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model training complete! Files saved: model.pkl & vectorizer.pkl")
