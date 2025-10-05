Overview
This project implements and compares sentiment analysis models on airline customer tweets, evaluating traditional machine learning (Logistic Regression with TF-IDF) against deep learning (Bidirectional LSTM) approaches. The goal is to classify customer sentiment into three categories: negative, neutral, and positive.

Key Features
Comprehensive EDA: 4+ visualizations including sentiment distribution, tweet length analysis, word clouds per sentiment, and top frequent words
Advanced Preprocessing: Text cleaning, tokenization, stopword removal, handling of Twitter-specific elements (@mentions, URLs, emoticons)
Feature Engineering: TF-IDF (n-grams 1-2, 10k features) for ML; Word embeddings (vocab 20k, max length 50) for DL
Class Imbalance Handling: SMOTE oversampling and class weights
Hyperparameter Optimization: Grid search for LR; learning rate experiments for BiLSTM
Robust Evaluation: Accuracy, F1-scores (weighted & macro), confusion matrices, ROC-AUC

Models Implemented

1. Logistic Regression (ML Baseline)
Features: TF-IDF vectorization (max 10k features, bigrams)
Architecture: Multi-class one-vs-rest with L2 regularization
Optimization: GridSearchCV (C=[0.1, 1, 5])
Best Parameters: C=5, penalty=L2, solver=liblinear

Results:
Accuracy: 77.9%
F1-Weighted: 77.0%
F1-Macro: 71.0%



3. Bidirectional LSTM (Deep Learning)

Architecture:

Embedding layer (20k vocab, 100 dims, random initialization)
Bidirectional LSTM (128 units, dropout 0.3, recurrent dropout 0.2)
Dense layer (64 units, ReLU)
Dropout (0.4)
Output layer (3 classes, softmax)


Training: Adam optimizer (lr=1e-4), categorical cross-entropy loss, early stopping (patience=3)
Results:

Accuracy: 75.2%
F1-Weighted: 74.1%
F1-Macro: 65.0%
ROC-AUC: 0.875

Installation
Clone the repository
cd sentiment-analysis-project

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

#run
python sentiment_analysis.py
