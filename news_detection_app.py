"""
Fully self-service Fake News Classifier.

Upload a CSV, pick which columns are the article text and the label
using dropdowns, click Train, see the results, then paste your own
text to get a prediction. No code editing required for a new dataset.

Run: streamlit run app.py
"""

import io
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@st.cache_resource
def ensure_nltk_data():
    checks = [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
    ]
    for pkg, path in checks:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")
st.title("📰 Fake News Classifier")
st.write(
    "Upload a labeled dataset, tell the app which columns to use, and it will "
    "clean the text, train a model, show you how well it performs, and let you "
    "test it on your own text — all on this page."
)

# ---------- Step 1: Upload ----------
st.header("1. Upload your dataset")
uploaded_file = st.file_uploader("CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{len(df)}** rows.")
    st.dataframe(df.head())

    # ---------- Step 2: Column mapping ----------
    st.header("2. Tell us which columns to use")
    columns = list(df.columns)
    text_col = st.selectbox("Which column contains the article text?", columns)
    label_col = st.selectbox(
        "Which column contains the real/fake label?", columns,
        index=min(1, len(columns) - 1),
    )

    unique_labels = df[label_col].dropna().unique().tolist()
    if len(unique_labels) != 2:
        st.warning(
            f"Expected exactly 2 unique label values in this column, found "
            f"{len(unique_labels)}: {unique_labels}. Pick a different column, "
            f"or clean the data so it only has two label values."
        )
    else:
        real_value = st.selectbox("Which value means REAL news?", unique_labels)
        fake_value = [v for v in unique_labels if v != real_value][0]
        st.caption(f"'{real_value}' → Real   ·   '{fake_value}' → Fake")

        # ---------- Step 3: Train ----------
        st.header("3. Train the model")
        test_size = st.slider("Test set size (%)", 10, 30, 15) / 100
        tune_hyperparams = st.checkbox(
            "Enable hyperparameter tuning (GridSearchCV) — slower, often improves accuracy",
            value=False,
        )

        if st.button("Train Model", type="primary"):
            work_df = df[[text_col, label_col]].dropna().copy()
            work_df["label"] = (work_df[label_col] == real_value).astype(int)

            with st.spinner("Cleaning text..."):
                work_df["clean_text"] = work_df[text_col].apply(clean_text)

            train_df, test_df = train_test_split(
                work_df, test_size=test_size, random_state=42, stratify=work_df["label"]
            )

            with st.spinner("Vectorizing and training..."):
                vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
                X_train = vectorizer.fit_transform(train_df["clean_text"])
                X_test = vectorizer.transform(test_df["clean_text"])

                model = LogisticRegression(max_iter=1000, class_weight="balanced")
                model.fit(X_train, train_df["label"])
                y_pred = model.predict(X_test)

            # Keep everything in session state so Step 4 below can use it
            st.session_state["model"] = model
            st.session_state["vectorizer"] = vectorizer
            st.session_state["real_value"] = real_value
            st.session_state["fake_value"] = fake_value

            st.success(f"Trained on {len(train_df)} rows, tested on {len(test_df)} rows.")
            st.metric("Accuracy", f"{accuracy_score(test_df['label'], y_pred) * 100:.1f}%")

            with st.expander("Full classification report"):
                st.text(classification_report(test_df["label"], y_pred, target_names=["Fake", "Real"]))

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                test_df["label"], y_pred, display_labels=["Fake", "Real"], ax=ax
            )
            st.pyplot(fig)

            # Let them download the trained model to reuse later without retraining
            buf = io.BytesIO()
            joblib.dump(
                {"model": model, "vectorizer": vectorizer, "real_value": real_value, "fake_value": fake_value},
                buf,
            )
            st.download_button("Download trained model (.pkl)", buf.getvalue(), file_name="fake_news_model.pkl")

# ---------- Step 4: Predict ----------
st.header("4. Try it on your own text")

if "model" not in st.session_state:
    st.info("Train a model above first, or load a previously downloaded one below.")
    pkl_file = st.file_uploader("Load a previously trained model (.pkl)", type=["pkl"], key="pkl_uploader")
    if pkl_file is not None:
        saved = joblib.load(pkl_file)
        st.session_state["model"] = saved["model"]
        st.session_state["vectorizer"] = saved["vectorizer"]
        st.session_state["real_value"] = saved["real_value"]
        st.session_state["fake_value"] = saved["fake_value"]
        st.success("Model loaded.")

if "model" in st.session_state:
    text = st.text_area("Paste news text here:", height=200)
    if st.button("Classify"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text)
            vec = st.session_state["vectorizer"].transform([cleaned])
            pred = st.session_state["model"].predict(vec)[0]
            proba = st.session_state["model"].predict_proba(vec)[0]
            confidence = max(proba)

            if pred == 1:
                st.success(f"✅ Predicted: **Real** ({st.session_state['real_value']})")
            else:
                st.error(f"🚩 Predicted: **Fake** ({st.session_state['fake_value']})")
            st.write(f"Confidence: {confidence * 100:.1f}%")
            st.progress(confidence)