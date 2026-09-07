import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Language Detection.csv")

print("Dataset shape:", df.shape)
print("Languages:", df["Language"].unique())


# ==========================================
# 2. TEXT CLEANING
# ==========================================

def clean(text):
    text = str(text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Cleaned_Text"] = df["Text"].apply(clean)


# ==========================================
# 3. INPUT / OUTPUT
# ==========================================

X = df["Cleaned_Text"]
y = df["Language"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 5),
    min_df=2,
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_vectorized, y_train)


# ==========================================
# 7. TEST MODEL
# ==========================================

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("MODEL RESULTS")
print("================================")
print("Accuracy:", accuracy)


# ==========================================
# 8. SAVE MODEL
# ==========================================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")


# ==========================================
# 9. TEST SOME EXAMPLES
# ==========================================

test_sentences = [
    "This is an English sentence.",
    "Hello, how are you today?",
    
    "Bonjour, comment allez-vous?",
    "Je suis très heureux aujourd'hui.",
    
    "Guten Morgen, wie geht es dir?",
    "Ich liebe dieses schöne Wetter.",
    
    "Hola, ¿cómo estás?",
    "Me gusta mucho este lugar.",
    
    "Привет, как дела?",
    "Я очень рад тебя видеть.",
    
    "Ciao, come stai?",
    "Mi piace molto questo posto.",
    
    "Olá, como você está?",
    "Eu gosto muito deste lugar.",
    
    "नमस्ते, आप कैसे हैं?",
    
    "こんにちは、お元気ですか？",
    
    "안녕하세요, 잘 지내세요?",
    
    "مرحبا، كيف حالك؟",
    
    "Buongiorno, come stai?",
]

print("\n================================")
print("SAMPLE PREDICTIONS")
print("================================")

for sentence in test_sentences:

    cleaned = clean(sentence)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    probability = model.predict_proba(vectorized).max()

    print(
        f"{sentence} --> {prediction} "
        f"({probability * 100:.2f}%)"
    )