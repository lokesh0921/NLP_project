# 🌐 NLP Language Detection Web Application

A complete **Natural Language Processing (NLP) based Language Detection
system** that identifies the language of text and can also accept an
image, extract its text using OCR, and then detect the language.

## Features

-   Direct text-based language detection
-   Image upload and OCR
-   Extracted-text display
-   TF-IDF character n-gram feature extraction
-   Logistic Regression multi-class classification
-   Confidence/probability display
-   Flask web application
-   17 supported languages

Supported languages: **English, Malayalam, Hindi, Tamil, Portuguese,
French, Dutch, Spanish, Greek, Russian, Danish, Italian, Turkish,
Swedish, Arabic, German, Kannada.**

> The dataset contains the labels `Portugeese` and `Sweedish`; these
> labels are retained for compatibility with the original dataset.

------------------------------------------------------------------------

## 🎯 Objectives

1.  Build an NLP-based language identification system.
2.  Apply text preprocessing and normalization.
3.  Extract character-level n-gram features.
4.  Apply TF-IDF feature weighting.
5.  Train a multi-class machine-learning classifier.
6.  Evaluate the model on unseen test data.
7.  Deploy the trained model through Flask.
8.  Extend the system with OCR so image text can also be classified.

------------------------------------------------------------------------

# 🧠 NLP Concepts Used

## 1. Introduction to NLP

Natural Language Processing enables computers to process human language.
This project is an NLP application because it receives human language
and predicts the language to which the text belongs.

``` text
Natural Language → Text Processing → Features → ML → Language
```

## 2. Levels of NLP

The project primarily operates at the **character/word-level statistical
processing and text-classification level**. It does not perform complete
syntactic, semantic, or pragmatic understanding.

## 3. Text Preprocessing

The `clean()` function performs:

-   punctuation removal
-   number removal
-   lowercasing
-   whitespace normalization

Example:

``` text
"Guten Morgen, Wie GEHT es Dir?"
              ↓
"guten morgen wie geht es dir"
```

## 4. Regular Expressions

Regular expressions are used to remove numerical patterns:

``` python
re.sub(r"\d+", "", text)
```

This demonstrates regex-based text cleaning and normalization.

## 5. Tokenization and Segmentation

Traditional NLP often tokenizes text into words. The final model does
**not use conventional word tokenization as its main representation**.
Instead, it uses character-level n-grams.

This is useful for language detection because languages have
characteristic spelling and character-combination patterns.

## 6. N-Grams

An n-gram is a sequence of consecutive units. The final model uses
character n-grams from length 1 to 5:

``` python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 5),
    min_df=2,
    sublinear_tf=True
)
```

For example, `German` can generate features such as `G`, `Ge`, `Ger`,
`Germ`, `Germa`, etc.

> The project uses n-grams as classification features; it is not
> implementing a generative n-gram language model.

## 7. Feature Extraction

Raw text must be converted into numerical features before machine
learning:

``` text
Raw Text → Preprocessing → Character N-Grams → TF-IDF → Feature Vector
```

## 8. TF-IDF

TF-IDF means **Term Frequency--Inverse Document Frequency**. In this
project, the terms are character n-grams rather than ordinary words.
TF-IDF gives different weights to features according to their frequency
and informativeness in the corpus.

## 9. Statistical NLP

Language detection is a statistical NLP task. The classifier learns
statistical relationships between character patterns and language labels
from the training corpus.

## 10. Text Classification

The central task is **17-class text classification**:

``` text
Input text → Classifier → One of 17 language labels
```

## 11. Logistic Regression

The final classifier is **Logistic Regression**. It receives the TF-IDF
feature vector and predicts one of the language classes.

``` text
Text → Cleaning → Character N-Grams → TF-IDF → Logistic Regression → Language
```

## 12. Probability / Confidence

The application uses:

``` python
model.predict_proba()
```

to obtain class probabilities. The highest probability is displayed as
the prediction confidence.

## 13. Train-Test Split

The project uses an 80/20 train-test split:

``` python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Approximately 80% of the data is used for training and 20% for testing.

## 14. Model Evaluation

The primary evaluation metric currently used is **accuracy**.

The final model achieved approximately:

``` text
98.79% test accuracy
```

Accuracy is:

``` text
Correct Predictions / Total Predictions × 100
```

------------------------------------------------------------------------

# 🖼️ OCR + NLP Integration

OCR means **Optical Character Recognition**. OCR itself is an
image/document-processing technique rather than an NLP technique. In
this project it acts as the input layer for the NLP system.

``` text
Image
  ↓
Tesseract OCR
  ↓
Extracted Text
  ↓
Text Preprocessing
  ↓
Character N-Grams
  ↓
TF-IDF
  ↓
Logistic Regression
  ↓
Detected Language
```

Example:

``` text
Image: "Bonjour, comment allez-vous?"
                 ↓ OCR
"Bonjour, comment allez-vous?"
                 ↓ NLP
French
```

------------------------------------------------------------------------

# 📚 Mumbai University NLP Topic Mapping

The University of Mumbai NLP curriculum includes NLP fundamentals,
levels of NLP, text preprocessing, regular expressions, n-grams, text
classification, feature extraction, statistical methods and evaluation.

  -----------------------------------------------------------------------
  NLP Topic               Used?                   How it appears in this
                                                  project
  ----------------------- ----------------------- -----------------------
  Introduction to NLP     ✅                      Language identification

  Generic NLP System      ✅                      Input → preprocessing →
                                                  features → classifier →
                                                  output

  Levels of NLP           ✅ Partially            Character/statistical
                                                  processing

  Text Preprocessing      ✅                      Cleaning and
                                                  normalization

  Regular Expressions     ✅                      Number removal

  Tokenization            ⚠️ Not directly         Character n-grams are
                                                  used instead

  Segmentation            ⚠️ Related              Character n-gram
                                                  segmentation

  Morphological Analysis  ❌                      Not required

  Lemmatization           ❌                      Not required

  Stemming                ❌                      Not required

  N-Grams                 ✅                      Character 1--5 grams

  N-Gram Language Model   ❌                      N-grams are features,
                                                  not a generative model

  Bag of Words            ❌ Final model          TF-IDF character
                                                  features are used

  TF-IDF                  ✅                      Feature weighting

  Statistical NLP         ✅                      Statistical language
                                                  classification

  Text Classification     ✅                      17-class language
                                                  detection

  Naive Bayes             ❌ Final model          Explored in the
                                                  original notebook, but
                                                  the deployed model uses
                                                  Logistic Regression

  Logistic Regression     ✅                      Final classifier

  Probability             ✅                      `predict_proba()`

  Model Evaluation        ✅                      Test accuracy

  Precision/Recall/F1     ❌ Currently            Future improvement

  Confusion Matrix        ❌ Currently            Future improvement

  POS Tagging             ❌                      Not required

  HMM                     ❌                      Not required

  Semantic Analysis       ❌                      No semantic
                                                  understanding is
                                                  required

  WordNet                 ❌                      Not required

  Word Sense              ❌                      Not required
  Disambiguation                                  

  Pragmatics              ❌                      Not required

  NLP Applications        ✅                      Language identification
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🔬 Complete NLP Pipeline

``` text
User Text
    ↓
Text Normalization
    ↓
Punctuation Removal
    ↓
Number Removal
    ↓
Lowercase Conversion
    ↓
Whitespace Normalization
    ↓
Character N-Grams (1–5)
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
17 Language Classes
    ↓
Language + Confidence
```

------------------------------------------------------------------------

# 🌐 Web Application Architecture

``` text
                     FRONTEND
                 HTML + CSS + JS
                       │
             ┌─────────┴─────────┐
             │                   │
         Text Input          Image Input
             │                   │
             │                  OCR
             │                   │
             │              Extracted Text
             │                   │
             └─────────┬─────────┘
                       ↓
                    FLASK
                       ↓
                 Text Cleaning
                       ↓
               vectorizer.pkl
                       ↓
                  model.pkl
                       ↓
                Language Result
```

### Flask endpoints

``` text
POST /predict
POST /predict-image
```

------------------------------------------------------------------------

# 📊 Dataset

The project uses `Language Detection.csv`.

``` text
10,337 text samples
17 language classes
```

Main columns:

``` text
Text
Language
```

------------------------------------------------------------------------

# 🤖 Model Training

``` text
Language Detection.csv
        ↓
Load Dataset
        ↓
Clean Text
        ↓
Train/Test Split
        ↓
TF-IDF Character Vectorizer
        ↓
Logistic Regression
        ↓
Evaluate
        ↓
Save Model
```

Generated files:

``` text
model.pkl
vectorizer.pkl
```

`model.pkl` stores the trained classifier. `vectorizer.pkl` stores the
fitted TF-IDF feature extractor. The same fitted vectorizer must be used
for new input so that features have the same representation as during
training.

------------------------------------------------------------------------

# 📁 Project Structure

``` text
NLP-Language-Detection-main/
│
├── Language Detection.csv
├── Language_Detection_Github_.ipynb
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── app.py
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── uploads/
```

  File                                 Purpose
  ------------------------------------ --------------------------------------
  `Language Detection.csv`             Training dataset
  `Language_Detection_Github_.ipynb`   Original NLP notebook
  `train_model.py`                     Trains and saves the final model
  `model.pkl`                          Saved Logistic Regression classifier
  `vectorizer.pkl`                     Saved TF-IDF vectorizer
  `app.py`                             Flask backend and prediction API
  `templates/index.html`               Web interface
  `static/style.css`                   Frontend styling
  `uploads/`                           Image-upload workspace

------------------------------------------------------------------------

# ⚙️ Installation

``` bash
pip3 install pandas scikit-learn flask pytesseract pillow
```

On macOS with Homebrew:

``` bash
brew install tesseract
brew install tesseract-lang
```

Check:

``` bash
tesseract --version
```

------------------------------------------------------------------------

# ▶️ Running the Project

### 1. Train the model

``` bash
python3 train_model.py
```

This creates:

``` text
model.pkl
vectorizer.pkl
```

### 2. Start Flask

``` bash
python3 app.py
```

### 3. Open the website

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

# 🧪 Example Inputs

### English

``` text
Hello, how are you today?
```

### French

``` text
Bonjour, comment allez-vous?
```

### German

``` text
Guten Morgen, wie geht es dir?
```

### Spanish

``` text
Hola, ¿cómo estás?
```

### Russian

``` text
Привет, как дела?
```

### Italian

``` text
Ciao, come stai?
```

------------------------------------------------------------------------

# 📈 Results

The final TF-IDF character n-gram + Logistic Regression model achieved
approximately:

``` text
98.79% test accuracy
```

The model was also manually tested with examples from several supported
languages.

------------------------------------------------------------------------

# ⚠️ Limitations

1.  The model is limited to the 17 languages in the training dataset.
2.  Very short or generic text can be harder to classify.
3.  OCR accuracy depends on image quality, font, resolution, orientation
    and installed OCR language data.
4.  The system performs language identification rather than semantic
    understanding.
5.  The current evaluation reports accuracy but does not yet show a full
    confusion matrix or per-class precision, recall and F1-score.
6.  Mixed-language text may produce a dominant-language prediction
    rather than separate predictions for each language.
7.  Confidence is a model probability, not a guarantee of real-world
    correctness.

------------------------------------------------------------------------

# 🚀 Future Improvements

-   Precision, Recall and F1-score
-   Confusion matrix
-   Per-language performance analysis
-   Better short-text handling
-   Language-specific OCR configuration
-   Automatic OCR language selection
-   Mixed-language detection
-   Drag-and-drop image upload
-   PDF text extraction
-   REST API deployment
-   Cloud deployment
-   Character CNN/RNN language identification
-   Transformer-based language identification

------------------------------------------------------------------------

# 🎓 Academic Relevance

This project demonstrates practical use of:

-   NLP fundamentals
-   Text preprocessing
-   Text normalization
-   Regular expressions
-   Character-level representation
-   N-grams
-   Feature extraction
-   TF-IDF
-   Statistical NLP
-   Multi-class text classification
-   Probability-based prediction
-   Train/test evaluation
-   NLP application development
-   OCR-to-NLP integration

It connects NLP theory with a working real-world application.

## Important distinction

The project does **not** implement every topic in the Mumbai University
NLP syllabus. Morphology, lemmatization, stemming, POS tagging, HMM,
semantic analysis, WordNet, Word Sense Disambiguation and pragmatics are
not required for this language-identification problem and are therefore
not claimed as implemented features.

------------------------------------------------------------------------

# 📚 References

-   University of Mumbai --- Natural Language Processing syllabus
-   Daniel Jurafsky & James H. Martin --- *Speech and Language
    Processing*
-   Christopher D. Manning & Hinrich Schütze --- *Foundations of
    Statistical Natural Language Processing*
-   Scikit-learn --- TF-IDF and Logistic Regression
-   Tesseract OCR

------------------------------------------------------------------------

# 👨‍💻 Project Summary

The core NLP idea is:

``` text
Human Language
      ↓
Preprocessing
      ↓
Character N-Grams
      ↓
TF-IDF
      ↓
Machine Learning
      ↓
Language Classification
```

The image extension adds:

``` text
Image
  ↓
OCR
  ↓
Text
  ↓
NLP Pipeline
  ↓
Language
```

Thus, the project combines **NLP preprocessing, statistical feature
extraction, machine-learning classification, OCR and web deployment**
into one complete application.
