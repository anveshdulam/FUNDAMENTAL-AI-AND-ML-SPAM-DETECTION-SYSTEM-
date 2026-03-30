# 📩 Spam Classifier — Ensemble ML

> *"Because one model's opinion is never enough."*

---

I built this because I was tired of seeing the same spam messages every day — fake lottery wins, UPI fraud traps, "your account will be blocked" nonsense — and I wanted to understand how a machine actually learns to tell the difference between a scam and a genuine message.

This is a fully terminal-based spam detector that doesn't rely on a single model. Instead, it trains **five different ML models at once**, lets them each vote on whether a message is spam or not, and gives you a confidence score based on how many of them agree. The more unanimous the vote, the more confident the result.

---

## 🤔 Why five models?

Every model has its own blind spots. Naive Bayes is great with word frequencies but misses context. SVM draws sharp boundaries but can overfit. Random Forest is robust but slow. Instead of picking one and hoping for the best, I made them vote — majority wins, and confidence reflects the agreement.

| Model | Why it's here |
|---|---|
| Naive Bayes | Classic, fast, surprisingly good at text |
| Logistic Regression | Strong linear baseline, very reliable |
| Decision Tree | Simple, interpretable, catches obvious patterns |
| Random Forest | 100 trees voting — hard to fool |
| Linear SVC (Calibrated) | High-accuracy margin-based classifier |

All five are trained using **TF-IDF vectorization** (top 500 features) and cached with `joblib` so after the first run, they load instantly every time.

---

## ✨ Features

- **Live classification** — type any message and get an instant SPAM / HAM verdict with confidence %
- **Batch mode** — classify a list of messages at once, or import from a `.txt` or `.csv` file
- **Spam keyword highlighting** — see exactly which words triggered the spam verdict
- **Session history** — review everything you classified in the current session
- **User feedback loop** — if the model gets it wrong, you can correct it and it learns from that correction on the next run
- **Results export** — batch results automatically saved to `results.csv`
- **Force retrain** — press `R` anytime to retrain all five models from scratch
- **CLI flag support** — use `--message` to classify a single message without entering the menu

---

## 🚀 Getting started

**Install dependencies:**
```bash
pip install scikit-learn pandas numpy joblib colorama
```

**Run the app:**
```bash
python live_prediction.py
```

**Classify a single message directly:**
```bash
python live_prediction.py --message "Congratulations! You've won a free iPhone. Click now!"
```

---

## 🗂️ Project structure

```
project/
│
├── live_prediction.py       ← main app (everything lives here)
├── feedback.csv             ← user corrections (auto-created)
├── results.csv              ← batch results export (auto-created)
└── saved_models/
    ├── models.joblib        ← trained models cache (auto-created)
    └── vectorizer.joblib    ← TF-IDF vectorizer cache (auto-created)
```

> The `saved_models/` folder and CSV files are created automatically on first run — you don't need to set anything up manually.

---

##  Training data

The dataset is a hand-curated mix of realistic spam and ham messages, with extra synthetic examples generated from templates to boost variety. It's specifically designed to handle **Indian spam patterns** — think fake UPI offers, Paytm fraud, Jio lucky draws, Aadhaar scam messages — alongside everyday student and personal messages like "bro did you finish the homework" and "when is the last date to submit the assignment."

The training data is split 80/20 for train/test, stratified to keep class balance consistent.

---

## 📊 How the verdict works

When you type a message, all five models predict independently. The final verdict is the **majority vote**, and the confidence score is:

```
confidence = (votes for winning class / 5) × 100
```

So if all 5 say SPAM → **100% confidence**. If 3 say SPAM and 2 say HAM → **60% confidence**. Simple, transparent, honest.

---

## 💬 Menu options

```
[1]  Classify a custom message
[2]  Run quick sample tests (6 built-in examples)
[3]  Batch mode (multiple messages or file import)
[4]  Exit + session summary
[5]  View session history
[R]  Force retrain all models
```

---

## 🛠️ Tech stack

`Python 3` · `scikit-learn` · `pandas` · `numpy` · `joblib` · `colorama`

---

##  Things I want to add next

- [ ] A simple web UI using Streamlit
- [ ] Support for multilingual spam (Hindi/Telugu SMS patterns)
- [ ] BERT or transformer-based model as a 6th ensemble member
- [ ] Email header parsing support (not just message body)

---

##  About

Built by **Dulam Anvesh Goud**
B.Tech CSE (AI & ML) — 1st Year | VIT Bhopal
[GitHub](https://github.com/anveshdulam) · [LinkedIn](https://www.linkedin.com/in/anvesh-goud-469124382) · [Kaggle](https://www.kaggle.com/anveshdulam)

> *"Building the future, one model at a time."*
