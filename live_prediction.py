"""Terminal-based spam classifier using an ensemble of classical ML models."""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

SPAM_LABEL = "SPAM"
HAM_LABEL = "HAM"

# File paths used by the app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODELS_PATH = os.path.join(MODEL_DIR, "models.joblib")
FEEDBACK_PATH = os.path.join(SCRIPT_DIR, "feedback.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.csv")

# Runtime session state
session_history = []          # list[dict]: {message, verdict, confidence}
session_spam_keywords = []    # all spam keywords observed in this run

# Base training dataset (manually curated examples)
DATASET = [
    # Spam
    ("Win a FREE iPhone now! Click the link to claim your prize", "spam"),
    ("Congratulations! You've won $5000. Call us immediately", "spam"),
    ("FREE entry to win fabulous cash prizes. Text WIN to 80085", "spam"),
    ("URGENT: Your account has been compromised. Verify now!", "spam"),
    ("You have been selected for a special loan offer. Apply now!", "spam"),
    ("Double your money in 24 hours with our investment plan!", "spam"),
    ("Exclusive deal just for you! Buy 1 get 10 FREE today only", "spam"),
    ("Call now to claim your free holiday to Bahamas!", "spam"),
    ("Limited time offer! Get 90% discount, click here", "spam"),
    ("You are a WINNER of our weekly lottery. Claim ASAP!", "spam"),
    ("Your mobile number has won 1 million dollars!", "spam"),
    ("Cheap meds delivered to your door. No prescription needed", "spam"),
    ("Make money fast working from home. Guaranteed income!", "spam"),
    ("FINAL NOTICE: Your bank account will be suspended!", "spam"),
    ("Earn 500 dollars per day with our proven system", "spam"),
    ("You have been pre-approved for a $10000 credit card!", "spam"),
    ("Free gift waiting for you! Claim before midnight tonight", "spam"),
    ("Hot singles near you are waiting. Click to meet them now", "spam"),
    ("Lose 20 pounds in 2 weeks guaranteed! Buy our pill now", "spam"),
    ("Your invoice is overdue! Pay now to avoid penalty charges", "spam"),
    ("Act now and receive a bonus cash reward in your account", "spam"),
    ("We noticed suspicious activity. Verify your identity here", "spam"),
    ("You qualify for a government tax refund. Claim yours now", "spam"),
    ("Lowest mortgage rates ever! Apply in minutes no credit check", "spam"),
    ("Congratulations! You've been selected for a free cruise to the Caribbean", "spam"),
    ("Get rich quick with our exclusive online trading platform. Sign up today!", "spam"),
    ("Your email account has been compromised. Click here to reset your password immediately", "spam"),
    ("Don't miss out on this once-in-a-lifetime opportunity to invest in our revolutionary new product. Act now!", "spam"),
    ("Congratulations! You've won a brand new car. Click here to claim your prize", "spam"),

    
    ("click here and get free money today no joke", "spam"),
    ("you won free recharge just send your number to us", "spam"),
    ("hi friend i have a job for you earn 5000 daily easy work", "spam"),
    ("send your bank details and we will transfer prize money", "spam"),
    ("free iphone 14 just fill this form quickly", "spam"),
    ("your account is hacked call this number now please", "spam"),
    ("get 1000 rupees free in your paytm just click this link", "spam"),
    ("you are selected for free laptop scheme apply now fast", "spam"),
    ("dear user your sim will stop working update info now", "spam"),
    ("earn money online just by watching videos signup free", "spam"),
    ("congratulation you got a prize of 50000 send otp to claim", "spam"),
    ("free data 2gb per day for 1 year click to activate now", "spam"),
    ("hello friend i found a way to make money fast wanna know", "spam"),
    ("your electricity bill is not paid pay now or connection cut", "spam"),
    ("win free shopping voucher worth 10000 click here today", "spam"),
    ("your google account will be deleted if you dont verify now", "spam"),
    ("get rich fast just invest 500 and earn 50000 in week", "spam"),
    ("free medicine home delivery no doctor needed order now", "spam"),
    ("your ration card is blocked update details to fix it now", "spam"),
    ("you have been chosen for government free money scheme apply", "spam"),
    ("dear customer your loan is approved collect it today only", "spam"),
    ("click this link to get free amazon gift card hurry up", "spam"),
    ("we are giving free bike to lucky winners you are one click", "spam"),
    ("your whatsapp is going to expire renew it by clicking here", "spam"),
    ("make 10000 weekly from mobile just 1 hour work daily", "spam"),
    ("hi this is bank calling your account has problem call back", "spam"),
    ("send us 200 rupees and we will send you 2000 back promise", "spam"),
    ("your number won in lucky draw collect prize call us now", "spam"),
    ("free online job work from home no experience needed apply", "spam"),
    ("urgent your debit card is blocked click link to unblock it", "spam"),
    ("get six pack abs in 7 days with our magic powder buy now", "spam"),
    ("you can earn 500 per hour by simple typing work from home", "spam"),
    ("free astrology prediction just give us your date of birth", "spam"),
    ("your aadhaar is linked with crime record call police helpline", "spam"),
    ("lottery result your ticket number won 1 crore claim fast", "spam"),
    ("dear winner please share your address to send prize to you", "spam"),
    ("click here get unlimited free recharge daily no cost at all", "spam"),
    ("we give loan without documents just send your photo and id", "spam"),
    ("your youtube channel will be deleted act fast click here", "spam"),
    ("hello this is amazon your order is stuck pay custom fee now", "spam"),
    ("just install this app and earn 300 rupees every hour easy", "spam"),
    ("free vaccine registration done pay 50 rs processing fee now", "spam"),
    ("your credit score is low we can fix it pay small fee now", "spam"),
    ("win free concert tickets just answer one simple question now", "spam"),
    ("earn from youtube without making video just buy our course", "spam"),
    ("free gold coin for first 100 users register now quickly", "spam"),
    ("your phone number is winner of 5 lakh in jio lucky draw", "spam"),
    ("click here to get free netflix subscription for whole year", "spam"),
    ("hi i am helping people make extra income interested message me", "spam"),
    ("you are pre selected for free government house scheme apply", "spam"),

    # Ham
    ("Hey, are you coming to class tomorrow morning?", "ham"),
    ("Can you please send me the assignment notes?", "ham"),
    ("What time is the practical lab today?", "ham"),
    ("Let's meet in the library after lunch to study", "ham"),
    ("Did you understand the last lecture on decision trees?", "ham"),
    ("Mom, I will be home late today. Don't wait for dinner", "ham"),
    ("Reminder: Submit your project report by Friday", "ham"),
    ("The professor postponed the test to next Monday", "ham"),
    ("Can you share the Python code we wrote in class?", "ham"),
    ("I am running 10 minutes late, please save my seat", "ham"),
    ("Our team meeting is rescheduled to 3 PM tomorrow", "ham"),
    ("Please review the document I sent and give feedback", "ham"),
    ("Are you free this weekend to work on the project?", "ham"),
    ("The canteen is closed today due to a function", "ham"),
    ("Happy birthday! Have a great day ahead", "ham"),
    ("The results for the mid-term exam are out now", "ham"),
    ("I have borrowed your book, will return it tomorrow", "ham"),
    ("Let me know if you want me to explain the algorithm", "ham"),
    ("Power cut in the hostel from 9 PM to 11 PM tonight", "ham"),
    ("Cricket match cancelled due to rain, see you tomorrow", "ham"),
    ("The wifi password in the lab has been changed today", "ham"),
    ("Can we reschedule our study group to Sunday instead?", "ham"),
    ("Your internship application has been received thank you", "ham"),
    ("Please bring your student ID card to the exam hall", "ham"),
    ("The seminar on machine learning is at 2 PM in hall B", "ham"),

    
    ("bro did you finish the homework or not tell me", "ham"),
    ("hey can you come online i need help with the code", "ham"),
    ("what did teacher say in class i was absent today", "ham"),
    ("yaar send me the notes i forgot to write them today", "ham"),
    ("when is the last date to submit the assignment tell me", "ham"),
    ("mom made food come home fast it will get cold", "ham"),
    ("bro the electricity went off in my area cant study now", "ham"),
    ("did you eat lunch or are you still in the lab", "ham"),
    ("i will be late to college today bus got delayed sorry", "ham"),
    ("can you explain what is overfitting i did not get it", "ham"),
    ("kal college hai ya nahi koi batao please", "ham"),
    ("hey the project deadline changed to next week i think", "ham"),
    ("are you bringing your laptop tomorrow for the viva", "ham"),
    ("sir said attendance is low please come regularly ok", "ham"),
    ("bro which book are you following for machine learning", "ham"),
    ("i scored 45 out of 50 in the test so happy today", "ham"),
    ("do you know the wifi password of the new lab block", "ham"),
    ("meeting cancelled for today we will do it tomorrow instead", "ham"),
    ("hey send me your phone number i lost all my contacts", "ham"),
    ("which chapter is coming in tomorrows test please tell me", "ham"),
    ("i am going to the library want to come with me now", "ham"),
    ("can you check my code i think there is a small bug", "ham"),
    ("today lunch was very bad in the canteen i am hungry", "ham"),
    ("bro i think i failed the exam feeling very sad now", "ham"),
    ("happy new year to you and your whole family from me", "ham"),
    ("did sir take attendance today i was sitting in back", "ham"),
    ("my laptop is not working can i use yours for sometime", "ham"),
    ("the class is shifted to room 204 from tomorrow onwards", "ham"),
    ("i finished the project just need to add some comments now", "ham"),
    ("when are you going home for holidays this time tell me", "ham"),
    ("sir gave extra assignment today very tough one it is bro", "ham"),
    ("are you coming to farewell party next week its on friday", "ham"),
    ("i got internship in a company so happy tell everyone", "ham"),
    ("hey do you have the question paper of last year exam", "ham"),
    ("bro can we meet at 5pm near the gate today please", "ham"),
    ("my marks are less than expected feeling a bit low today", "ham"),
    ("today sir explained neural networks very nicely in class", "ham"),
    ("please save me a seat in the front row tomorrow class", "ham"),
    ("do you want to form a group for the mini project work", "ham"),
    ("i returned your notes sorry for the late return my friend", "ham"),
    ("the viva is rescheduled to thursday check college portal", "ham"),
    ("bro are you on campus right now i need urgent help", "ham"),
    ("today no class period free lets go to canteen together", "ham"),
    ("i will call you after reaching home just wait a bit", "ham"),
    ("can you share the ppt of todays lecture i missed it", "ham"),
    ("hey did you register for the hackathon event or not yet", "ham"),
    ("results came out i passed in all subjects so relieved now", "ham"),
    ("good morning hope you slept well see you in class today", "ham"),
    ("my project demo went well sir liked it a lot today", "ham"),
    ("bro where are you we are all waiting at the lab now", "ham"),
]


def build_extra_dataset(spam_count=150, ham_count=150):
    """Create extra synthetic samples so both classes have more variety."""
    spam_templates = [
        "urgent alert your {account} needs verification now click {link}",
        "congratulations you won {amount} cash reward claim with code {code}",
        "limited offer get {discount}% discount on {product} today only",
        "free {product} for first {users} users register immediately",
        "your {service} will be blocked in {minutes} minutes update details now",
        "earn {amount} per day from home with no experience join now",
        "dear customer pay {fee} processing fee to release your prize",
        "exclusive investment plan grow money {multiplier}x in one week",
        "bank notice suspicious login detected confirm otp to secure account",
        "click to activate free subscription for {service} valid for {days} days",
    ]

    ham_templates = [
        "can we meet at {time} in the {place} to discuss project",
        "please share the notes for {subject} i missed that class",
        "reminder our team meeting is on {day} at {time}",
        "i uploaded the assignment for {subject} please review once",
        "class for {subject} shifted to room {room} tomorrow",
        "i will reach campus by {time} save me a seat",
        "did you complete the lab record for {subject} this week",
        "let us revise {topic} together after lunch in {place}",
        "faculty said viva starts on {day} bring your id card",
        "thanks for helping me with {topic} your explanation worked",
    ]

    accounts = ["bank account", "email", "wallet", "upi", "sim card", "profile"]
    links = ["this link", "secure portal", "official form", "verification page"]
    amounts = ["5000", "10000", "25000", "50000", "1 lakh"]
    discounts = ["40", "50", "60", "70", "80", "90"]
    products = ["smartphone", "laptop", "voucher", "headphones", "gift card", "tablet"]
    users = ["50", "100", "200", "500"]
    services = ["whatsapp", "banking", "netflix", "sim", "wallet", "mail"]
    minutes = ["10", "15", "30", "45", "60"]
    fees = ["49", "99", "199", "299"]
    multipliers = ["2", "3", "5", "10"]
    days = ["3", "7", "15", "30"]
    codes = ["A12", "B77", "C91", "D45", "X09", "P30"]

    times = ["9am", "10am", "11am", "1pm", "3pm", "5pm"]
    places = ["library", "lab", "canteen", "classroom", "seminar hall"]
    subjects = ["python", "machine learning", "dbms", "os", "maths"]
    days_list = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    rooms = ["101", "204", "305", "B12", "C21"]
    topics = ["classification", "overfitting", "regression", "nlp", "feature scaling"]

    extra = []

    for i in range(spam_count):
        tpl = spam_templates[i % len(spam_templates)]
        msg = tpl.format(
            account=accounts[i % len(accounts)],
            link=links[i % len(links)],
            amount=amounts[i % len(amounts)],
            code=codes[i % len(codes)],
            discount=discounts[i % len(discounts)],
            product=products[i % len(products)],
            users=users[i % len(users)],
            service=services[i % len(services)],
            minutes=minutes[i % len(minutes)],
            fee=fees[i % len(fees)],
            multiplier=multipliers[i % len(multipliers)],
            days=days[i % len(days)],
        )
        extra.append((f"{msg} ref{i+1}", "spam"))

    for i in range(ham_count):
        tpl = ham_templates[i % len(ham_templates)]
        msg = tpl.format(
            time=times[i % len(times)],
            place=places[i % len(places)],
            subject=subjects[i % len(subjects)],
            day=days_list[i % len(days_list)],
            room=rooms[i % len(rooms)],
            topic=topics[i % len(topics)],
        )
        extra.append((f"{msg} note{i+1}", "ham"))

    return extra


DATASET.extend(build_extra_dataset(spam_count=350, ham_count=350))

def load_feedback():
    """Load corrected examples from feedback.csv and append to DATASET."""
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        count = 0
        for _, row in df.iterrows():
            msg = str(row.get("message", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            if msg and label in ("spam", "ham"):
                DATASET.append((msg, label))
                count += 1
        return count
    except Exception:
        return 0


def save_feedback(message, correct_label):
    """Append a single corrected example to feedback.csv."""
    file_exists = os.path.exists(FEEDBACK_PATH)
    df_new = pd.DataFrame([{"message": message, "label": correct_label}])
    df_new.to_csv(FEEDBACK_PATH, mode="a", header=not file_exists, index=False)


# Load feedback examples during startup
feedback_loaded = load_feedback()

def train_all_models():
    """Train all models from scratch on the full DATASET."""
    messages = [d[0] for d in DATASET]
    labels = [1 if d[1] == "spam" else 0 for d in DATASET]

    vec = TfidfVectorizer(stop_words="english", lowercase=True, max_features=500)
    X = vec.fit_transform(messages)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes         ": MultinomialNB(),
        "Logistic Regression ": LogisticRegression(max_iter=1000),
        "Decision Tree       ": DecisionTreeClassifier(random_state=42),
        "Random Forest       ": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM                 ": CalibratedClassifierCV(LinearSVC(max_iter=1000), cv=3),
    }

    trained_models = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, zero_division=0)
        trained_models[name] = {"model": m, "acc": acc, "f1": f1}

    # Fit once on train/test split for metrics, then fit on full data for live usage.
    for name in trained_models:
        trained_models[name]["model"].fit(X, y)

    return trained_models, vec


def save_models(models_dict, vectorizer):
    """Save trained models and vectorizer to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(models_dict, MODELS_PATH)


def load_models():
    """Load models and vectorizer from disk. Returns (models_dict, vectorizer) or (None, None)."""
    if os.path.exists(MODELS_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            models_dict = joblib.load(MODELS_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return models_dict, vectorizer
        except Exception:
            return None, None
    return None, None


def get_models(force_retrain=False):
    """Return ready-to-use models — load from disk or train fresh."""
    if not force_retrain:
        models_dict, vectorizer = load_models()
        if models_dict is not None:
            print(f"\n  {Fore.GREEN}✅ Loaded saved models from disk.{Style.RESET_ALL}")
            if feedback_loaded:
                print(f"  {Fore.YELLOW}📝 {feedback_loaded} feedback examples were loaded — retraining to include them...{Style.RESET_ALL}")
                # Feedback changed the dataset, so refresh saved models.
            else:
                return models_dict, vectorizer

    loading_bar("🔧 Preparing dataset and training 5 ML models...", steps=25, delay=0.05)
    models_dict, vectorizer = train_all_models()
    save_models(models_dict, vectorizer)
    print(f"  {Fore.GREEN}💾 Models saved to disk for future runs.{Style.RESET_ALL}")
    return models_dict, vectorizer


def divider(char="─", width=60):
    print("  " + char * width)

def banner():
    print("\n" + "═"*64)
    print(" SPAM CLASSIFIER — LIVE PREDICTION")
    print("AI/ML Project | BY: DULAM ANVESH GOUD (25BAI10595)")
    print("═"*64)

def loading_bar(task, steps=20, delay=0.04):
    """Render a small terminal spinner while long tasks run."""
    print(f"\n  {task}")
    animation = "|/-\\"
    for i in range(steps):
        time.sleep(delay)
        sys.stdout.write(f"\r  {animation[i % len(animation)]} Working...")
        sys.stdout.flush()
    print("\r  ✓ Done!            ")


def color_verdict(verdict):
    """Return a colorized verdict string."""
    if verdict == SPAM_LABEL:
        return f"{Fore.RED}{SPAM_LABEL}{Style.RESET_ALL}"
    return f"{Fore.GREEN}{HAM_LABEL}{Style.RESET_ALL}"


def score_bar(confidence_pct, width=10):
    """Return a compact visual confidence bar (for example: ████████░░)."""
    filled = round(confidence_pct / 100 * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def get_top_spam_keywords(message, vectorizer, n=5):
    """Return the top-n TF-IDF keywords from the message that overlap with the
    vectorizer's vocabulary (i.e., most 'important' words in this message)."""
    vec = vectorizer.transform([message])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = vec.toarray().flatten()
    top_idx = scores.argsort()[::-1]
    keywords = []
    for idx in top_idx:
        if scores[idx] > 0:
            keywords.append(feature_names[idx])
        if len(keywords) >= n:
            break
    return keywords


def predict_single(message, models_dict, vectorizer):
    """Run all models on a single message and return voting details."""
    vec = vectorizer.transform([message])
    votes_spam = 0
    details = []

    for name, info in models_dict.items():
        model = info["model"]
        pred = model.predict(vec)[0]
        label = SPAM_LABEL if pred == 1 else HAM_LABEL
        if pred == 1:
            votes_spam += 1

        model_conf = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            model_conf = proba[pred] * 100
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(vec)[0]
            # Convert SVM margin score to probability-like confidence.
            model_conf = 1.0 / (1.0 + np.exp(-dec)) * 100
            if pred == 0:
                model_conf = 100 - model_conf
        else:
            model_conf = 100.0

        details.append((name.strip(), label, info["acc"], info["f1"], model_conf))

    total = len(models_dict)
    final = SPAM_LABEL if votes_spam > total // 2 else HAM_LABEL
    confidence = (votes_spam / total * 100) if final == SPAM_LABEL else ((total - votes_spam) / total * 100)

    return final, confidence, votes_spam, total, details


def display_result(message, final, confidence, votes_spam, total, details, vectorizer):
    divider("═")
    print(f"\n  📨  Message : \"{message}\"")
    divider()

    header = f"  {'Model':<28} {'Prediction':^12} {'Accuracy':^10} {'F1-Score':^10} {'Confidence':^12} {'Bar':^12}"
    print(f"\n{header}")
    divider("-")
    for name, label, acc, f1, model_conf in details:
        icon = "🚨" if label == SPAM_LABEL else "✅"
        colored_label = f"{Fore.RED}{label}{Style.RESET_ALL}" if label == SPAM_LABEL else f"{Fore.GREEN}{label}{Style.RESET_ALL}"
        bar = score_bar(model_conf)
        print(f"  {name:<28} {icon} {colored_label:<18} {acc*100:>8.1f}%  {f1*100:>8.1f}%  {model_conf:>7.1f}%  {bar}")
    divider("-")

    print(f"\n  📊  Voting Result : {votes_spam}/{total} models say SPAM")
    print(f"  🔒  Confidence    : {confidence:.1f}%")
    divider()

    if final == SPAM_LABEL:
        print(f"  {Fore.RED}🚨  FINAL VERDICT :  S P A M  ⚠️{Style.RESET_ALL}")
        print(f"  {Fore.RED}⚠️   This message looks suspicious. Avoid links and attachments.{Style.RESET_ALL}")
        keywords = get_top_spam_keywords(message, vectorizer, n=5)
        if keywords:
            session_spam_keywords.extend(keywords)
            kw_str = ", ".join(keywords)
            print(f"  {Fore.YELLOW}🔑  Top spam signals: {kw_str}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.GREEN}✅  FINAL VERDICT :  H A M  (Safe Message) 😊{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}👍  This message appears to be legitimate.{Style.RESET_ALL}")
    divider("═")


def ask_feedback(message, predicted_label):
    """Ask the user whether the prediction was correct. If not, save correction."""
    fb = input(f"\n  Was this prediction correct? (y/n): ").strip().lower()
    if fb == "n":
        correct = "ham" if predicted_label == SPAM_LABEL else "spam"
        save_feedback(message, correct)
        print(f"  {Fore.YELLOW}📝 Feedback saved! The correct label '{correct}' will be used in future training.{Style.RESET_ALL}")


def add_session_entry(message, verdict, confidence):
    """Store one prediction result in session history."""
    session_history.append({"message": message, "verdict": verdict, "confidence": confidence})


SAMPLES = [
    ("Win a FREE iPhone now! Click to claim your prize",     "spam"),
    ("Congratulations! You've won $5000. Call immediately",  "spam"),
    ("URGENT: Your bank account has been suspended!",        "spam"),
    ("Hey, are you coming to class tomorrow?",               "ham"),
    ("Can you send me the notes from today's lecture?",      "ham"),
    ("The professor postponed the exam to Monday",           "ham"),
]

def run_sample_tests(models_dict, vectorizer):
    print("\n" + "═"*64)
    print("  📋  QUICK SAMPLE TEST — 6 Preloaded Messages")
    print("═"*64)
    correct = 0
    for msg, true_label in SAMPLES:
        final, conf, vs, total, _ = predict_single(msg, models_dict, vectorizer)
        predicted = final.lower()
        status = f"{Fore.GREEN}✅ Correct{Style.RESET_ALL}" if predicted == true_label else f"{Fore.RED}❌ Wrong{Style.RESET_ALL}"
        colored_final = color_verdict(final)
        print(f"  {status}  [{conf:4.0f}% conf]  '{msg[:45]}...' → {colored_final}")
        if predicted == true_label:
            correct += 1
        add_session_entry(msg, final, conf)
        if final == SPAM_LABEL:
            kw = get_top_spam_keywords(msg, vectorizer, n=5)
            session_spam_keywords.extend(kw)
    print(f"\n  Sample Accuracy: {correct}/{len(SAMPLES)} correct")
    divider("═")


def import_messages_from_file():
    """Prompt user for a .txt or .csv file path and return list of messages."""
    path = input("\n  Enter file path (.txt or .csv): ").strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"  {Fore.RED}[!] File not found: {path}{Style.RESET_ALL}")
        return []
    ext = os.path.splitext(path)[1].lower()
    messages = []
    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(line)
        elif ext == ".csv":
            df = pd.read_csv(path)
            if "message" not in df.columns:
                print(f"  {Fore.RED}[!] CSV must have a column named 'message'.{Style.RESET_ALL}")
                return []
            messages = df["message"].dropna().astype(str).str.strip().tolist()
            messages = [m for m in messages if m]
        else:
            print(f"  {Fore.RED}[!] Unsupported file type '{ext}'. Use .txt or .csv.{Style.RESET_ALL}")
            return []
    except Exception as e:
        print(f"  {Fore.RED}[!] Error reading file: {e}{Style.RESET_ALL}")
        return []
    print(f"  {Fore.GREEN}✅ Loaded {len(messages)} messages from file.{Style.RESET_ALL}")
    return messages


def export_results_csv(results):
    """Ask user if they want to export batch results; if yes, save to results.csv."""
    ans = input("\n  Export results to CSV? (y/n): ").strip().lower()
    if ans == "y":
        df = pd.DataFrame(results, columns=["message", "verdict", "confidence"])
        df.to_csv(RESULTS_PATH, index=False)
        print(f"  {Fore.GREEN}💾 Results exported to {RESULTS_PATH}{Style.RESET_ALL}")


def print_session_summary():
    """Print a summary of the session when user exits."""
    total = len(session_history)
    if total == 0:
        print(f"\n  {Fore.YELLOW}📊 No messages were classified in this session.{Style.RESET_ALL}")
        return

    spam = sum(1 for h in session_history if h["verdict"] == SPAM_LABEL)
    ham = total - spam
    pct = (spam / total) * 100

    print("\n" + "═"*64)
    print(f"  {Fore.CYAN}📊  SESSION SUMMARY{Style.RESET_ALL}")
    print("═"*64)
    print(f"  Total messages classified : {total}")
    print(f"  {Fore.RED}🚨 SPAM{Style.RESET_ALL}                    : {spam}")
    print(f"  {Fore.GREEN}✅ HAM{Style.RESET_ALL}                     : {ham}")
    print(f"  Spam percentage           : {pct:.1f}%")

    if session_spam_keywords:
        counter = Counter(session_spam_keywords)
        top3 = counter.most_common(3)
        kw_str = ", ".join([w for w, _ in top3])
        print(f"  {Fore.YELLOW}🔑 Top 3 spam keywords{Style.RESET_ALL}      : {kw_str}")

    divider("═")


def view_session_history():
    """Print all messages classified so far in this session."""
    if not session_history:
        print(f"\n  {Fore.YELLOW}[!] No messages classified yet.{Style.RESET_ALL}")
        return

    print("\n" + "═"*64)
    print(f"  {Fore.CYAN}📜  SESSION HISTORY{Style.RESET_ALL}")
    print("═"*64)
    print(f"  {'#':<5} {'Verdict':<10} {'Conf':>7}   Message")
    divider("-")
    for idx, h in enumerate(session_history, 1):
        v = color_verdict(h["verdict"])
        print(f"  {idx:<5} {v:<18} {h['confidence']:>5.1f}%   {h['message'][:50]}")
    divider("═")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spam Classifier — Ensemble ML (Terminal)"
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        default=None,
        help='Classify a single message directly, e.g.  --message "Win free prize now"'
    )
    return parser.parse_args()


def print_model_metrics(models_dict):
    """Print model-wise accuracy and F1 in a compact table-like format."""
    print("\n  Model Performance:")
    for name, info in models_dict.items():
        print(f"     • {name.strip():<25}  Accuracy: {info['acc']*100:.1f}%   F1: {info['f1']*100:.1f}%")


def main():
    args = parse_args()

    banner()

    models_dict, vectorizer = get_models(force_retrain=False)

    print(f"\n  {Fore.GREEN}✅ Training complete. All 5 models are ready.{Style.RESET_ALL}")
    if feedback_loaded:
        print(f"  {Fore.YELLOW}📝 Incorporated {feedback_loaded} feedback examples from feedback.csv{Style.RESET_ALL}")
    print_model_metrics(models_dict)

    if args.message:
        msg = args.message.strip()
        if msg:
            final, conf, vs, total, details = predict_single(msg, models_dict, vectorizer)
            display_result(msg, final, conf, vs, total, details, vectorizer)
            add_session_entry(msg, final, conf)
        return

    while True:
        print("\n" + "─"*64)
        print("  MENU")
        print("  ─────────────────────────────────")
        print("  [1]  Classify a custom message")
        print("  [2]  Run quick sample tests (6 examples)")
        print("  [3]  Batch mode (classify multiple messages)")
        print("  [4]  Exit")
        print(f"  [5]  View session history")
        print(f"  [R]  Force retrain models")
        print("─"*64)

        choice = input("\n  Enter choice (1/2/3/4/5/R): ").strip().upper()

        if choice == "1":
            print("\n  Type your message below (type 'back' to return to menu):")
            while True:
                msg = input("\n  📩  Message: ").strip()
                if msg.lower() in ("back", "b", "menu"):
                    break
                if not msg:
                    print("  [!] Message cannot be empty.")
                    continue
                final, conf, vs, total, details = predict_single(msg, models_dict, vectorizer)
                display_result(msg, final, conf, vs, total, details, vectorizer)
                add_session_entry(msg, final, conf)
                ask_feedback(msg, final)
                again = input("\n  Classify another message? (y/n): ").strip().lower()
                if again != "y":
                    break

        elif choice == "2":
            run_sample_tests(models_dict, vectorizer)

        elif choice == "3":
            print("\n  BATCH MODE — Enter messages one by one, OR import from file.")
            print("  Type 'import' to load from .txt/.csv, or start typing messages.")
            print("  Type 'done' when finished.\n")
            batch = []
            i = 1
            while True:
                msg = input(f"  Message {i}: ").strip()
                if msg.lower() == "done":
                    break
                if msg.lower() == "import":
                    imported = import_messages_from_file()
                    batch.extend(imported)
                    i += len(imported)
                    continue
                if msg:
                    batch.append(msg)
                    i += 1

            if not batch:
                print("  [!] No messages entered.")
            else:
                print(f"\n  ═══ BATCH RESULTS ({len(batch)} messages) ═══")
                spam_count = 0
                export_data = []
                for idx, msg in enumerate(batch, 1):
                    final, conf, vs, total, _ = predict_single(msg, models_dict, vectorizer)
                    colored_icon = f"{Fore.RED}🚨 SPAM{Style.RESET_ALL}" if final == "SPAM" else f"{Fore.GREEN}✅ HAM {Style.RESET_ALL}"
                    print(f"  {idx}. {colored_icon}  ({conf:.0f}% conf)  —  {msg[:50]}")
                    if final == SPAM_LABEL:
                        spam_count += 1
                        kw = get_top_spam_keywords(msg, vectorizer, n=5)
                        if kw:
                            session_spam_keywords.extend(kw)
                            print(f"     {Fore.YELLOW}🔑 Top spam signals: {', '.join(kw)}{Style.RESET_ALL}")
                    add_session_entry(msg, final, conf)
                    export_data.append((msg, final, f"{conf:.1f}%"))

                print(f"\n  Summary: {Fore.RED}{spam_count} SPAM{Style.RESET_ALL}, "
                      f"{Fore.GREEN}{len(batch)-spam_count} HAM{Style.RESET_ALL} out of {len(batch)} messages")
                divider("═")
                export_results_csv(export_data)

        elif choice == "4":
            print_session_summary()
            print(f"\n  👋 Thank you for using Spam Classifier. Goodbye!\n")
            break

        elif choice == "5":
            view_session_history()

        elif choice == "R":
            print(f"\n  {Fore.YELLOW}🔄 Force retraining all models...{Style.RESET_ALL}")
            models_dict, vectorizer = get_models(force_retrain=True)
            print(f"\n  {Fore.GREEN}✅ Retrain complete. All 5 models updated.{Style.RESET_ALL}")
            print_model_metrics(models_dict)

        else:
            print("  [!] Invalid choice. Please enter 1, 2, 3, 4, 5, or R.")


if __name__ == "__main__":
    main()