# Fake News Classifier — No-Code Streamlit App

A single-file Streamlit app for classifying news articles as **Real** or **Fake**.
Upload any labeled dataset, pick your columns from dropdowns, train a model,
and test it — all from the browser. No code editing required.

---

## Features

- Upload any CSV dataset (your own column names are fine)
- Pick which column is the article text and which is the label via dropdowns
- Automatic text cleaning (lowercasing, URL/HTML stripping, stopword removal, lemmatization)
- TF-IDF vectorization + Logistic Regression classifier
- Optional hyperparameter tuning (GridSearchCV) for better accuracy
- Accuracy, full classification report, and a confusion matrix shown after training
- Type or paste any text and get an instant Real/Fake prediction with a confidence score
- Download the trained model to reuse later without retraining

---

## Requirements

- Python 3.9+
- See `requirements.txt` for exact packages

## Installation

```bash
pip install -r requirements.txt
```

The first run will also download a few small NLTK data files (stopwords,
tokenizer, lemmatizer) automatically — this only happens once.

## Running the app

```bash
streamlit run no_code_fake_news_app.py
```

This opens the app in your default browser (usually `http://localhost:8501`).

---

## How to use it

1. **Upload your dataset** — a CSV file with at least one text column and one label column.
2. **Pick your columns** — select which column has the article text, which has the label, and which label value means "Real" (everything else is treated as "Fake").
   > Your label column must have exactly two unique values (e.g. `REAL`/`FAKE`, `1`/`0`, `true`/`false`). If it has more, pick a different column or clean your data first.
3. **(Optional) Enable hyperparameter tuning** — checks a range of regularization strengths via 5-fold cross-validation and picks the best one. More accurate, but slower to train.
4. **Click Train Model** — the app cleans the text, builds TF-IDF features, trains Logistic Regression, and shows you:
   - Overall accuracy
   - A full precision/recall/F1 classification report
   - A confusion matrix
5. **Download the trained model** (optional) — saves a `.pkl` file you can re-upload later instead of retraining.
6. **Try it yourself** — paste any news text into the box at the bottom and click Classify to see a Real/Fake prediction with a confidence score.

---

## Notes & limitations

- This app keeps everything in the browser session (`st.session_state`). Refreshing the page clears the trained model — re-upload the `.pkl` you downloaded, or retrain.
- Works best on datasets of at least a few hundred labeled rows per class; very small datasets will train but may not generalize well.
- The model only knows what your dataset teaches it — if your data mostly reflects one time period, topic, or source, predictions on very different text may be less reliable.
- This is a standalone app and does not depend on any other files — just this one script and its requirements.
