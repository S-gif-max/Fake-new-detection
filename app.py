import streamlit as st
import pandas as pd
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -------------------------------------------
# Auto-train model if not found
# -------------------------------------------
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = "news_dataset_10000.csv"

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.write("🔄 Training model for the first time...")
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset file '{DATA_FILE}' not found. Please add it to the project folder.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    st.success("✅ Model trained and saved successfully!")

# Load model and vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# -------------------------------------------
# Streamlit App Config
# -------------------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# -------------------------------------------
# Custom CSS Styling
# -------------------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f4ff, #e3f2fd);
        background-attachment: fixed;
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
    }

    .main-title {
        text-align: center;
        color: #1a237e;
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 40px;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #5c6bc0;
        font-size: 16px;
        box-shadow: 0px 3px 10px rgba(92,107,192,0.2);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #3949ab, #5c6bc0);
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 220px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
        margin: 20px auto;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1a237e, #283593);
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0px 6px 18px rgba(26,35,126,0.4);
    }

    .result-box {
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        color: white;
        animation: fadeIn 0.6s ease-in-out;
    }
    .fake { background-color: #e57373; }
    .real { background-color: #81c784; }

    @keyframes fadeIn {
        from {opacity: 0; transform: scale(0.9);}
        to {opacity: 1; transform: scale(1);}
    }

    footer {
        text-align: center;
        color: #444;
        font-size: 14px;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------
# Page Layout
# -------------------------------------------
st.markdown("<h1 class='main-title'>📰 Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter text manually or analyze live news from the internet.</p>", unsafe_allow_html=True)

# -------------------------------------------
# User Input Section
# -------------------------------------------
text_input = st.text_area("📝 Paste your news content here 👇", height=200, placeholder="Type or paste news text...")

if st.button("Analyze Manually"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        input_vec = vectorizer.transform([text_input])
        prediction = model.predict(input_vec)[0]

        if prediction.lower() == "fake":
            st.markdown("<div class='result-box fake'>🚨 This news appears to be **FAKE**.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box real'>✅ This news appears to be **REAL**.</div>", unsafe_allow_html=True)

# -------------------------------------------
# Real-Time News Analysis (NewsData.io)
# -------------------------------------------
st.markdown("---")
st.subheader("🌍 Real-Time News Analysis")

api_key = "pub_33ab1fef9fc54983b3b1b49de38d1a39"  # 🔹 Replace with your actual key

topic = st.text_input("Enter a topic (e.g. politics, sports, technology):", "technology")

if st.button("Fetch and Analyze Live News"):
    with st.spinner("Fetching latest news..."):
        url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={topic}&language=en"
        response = requests.get(url)
        data = response.json()

        if "results" in data:
            for article in data["results"][:5]:
                title = article.get("title", "")
                desc = article.get("description", "")
                combined_text = f"{title} {desc}"

                if combined_text.strip() == "":
                    continue

                input_vec = vectorizer.transform([combined_text])
                prediction = model.predict(input_vec)[0]

                st.markdown(f"### 🗞️ {title}")
                st.write(desc if desc else "_No description available_")

                if prediction.lower() == "fake":
                    st.markdown("<div class='result-box fake'>🚨 Predicted: **FAKE NEWS**</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-box real'>✅ Predicted: **REAL NEWS**</div>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.error("❌ No news found or API limit reached.")

# Footer
st.markdown("<footer>Developed with ❤️ using Streamlit, Machine Learning & NewsData.io</footer>", unsafe_allow_html=True)
