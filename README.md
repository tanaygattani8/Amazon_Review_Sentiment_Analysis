# Amazon Review Sentiment Analysis

An NLP pipeline for classifying Amazon product reviews as Positive or Negative using machine learning, with end-to-end data cleaning, feature engineering, and model evaluation.

## Overview

This project analyzes a large dataset of Amazon product reviews (10,000+ raw rows) and builds a sentiment classification model. The pipeline covers raw data ingestion, noise removal, TF-IDF vectorization, and model training, producing a clean, production-ready NLP classifier.

## Features

- **Large-Scale Data Cleaning** - Processes 10,000 raw reviews down to 919 high-quality labeled samples through aggressive filtering and deduplication.
- **Advanced Text Preprocessing** - Implements lowercasing, HTML tag removal, stopword filtering, and lemmatization for clean feature extraction.
- **TF-IDF Vectorization** - Converts cleaned text into numerical feature vectors using Term Frequency-Inverse Document Frequency weighting.
- **Multi-Model Evaluation** - Trains and benchmarks multiple classifiers (Logistic Regression, Naive Bayes, SVM) to find the best performer.
- **Confusion Matrix and Metrics** - Full evaluation including accuracy, precision, recall, F1-score, and a visual confusion matrix.
- **Sentiment Distribution Analysis** - Visualizes the distribution of positive vs. negative reviews in the dataset.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| NLP and ML | scikit-learn, NLTK, TF-IDF |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Environment | Jupyter Notebook |

## Project Structure

```
Amazon_Review_Sentiment_Analysis/
├── Sentiment_Analysis.ipynb
└── README.md
```

## Pipeline Walkthrough

**1. Data Loading and Exploration**

Loads the raw Amazon review dataset (10,000 rows x 14 columns). Performs initial EDA including shape inspection, null value checks, and class distribution analysis.

**2. Data Cleaning**

Removes duplicates and null reviews. Filters down to a focused, high-quality subset (919 rows x 17 columns). Strips HTML tags and special characters from review text.

**3. Text Preprocessing**

Converts all text to lowercase, tokenizes reviews, removes English stopwords using NLTK, and applies lemmatization to reduce words to their base forms.

**4. Feature Engineering with TF-IDF**

Transforms the cleaned corpus using TfidfVectorizer. Captures word importance across the entire review corpus to create meaningful numerical representations.

**5. Model Training and Evaluation**

Splits data into train/test sets (80/20). Trains multiple classifiers and evaluates them with accuracy, precision, recall, F1-score, and a confusion matrix heatmap.

## Dataset

| Detail | Info |
|---|---|
| Source | Amazon Product Reviews |
| Raw Size | 10,000 rows x 14 columns |
| Cleaned Size | 919 rows x 17 columns |
| Target Label | Sentiment (Positive / Negative) |
| Task Type | Binary Text Classification |

## Running Locally

**Prerequisites:** Python 3.9+, Jupyter Notebook

```bash
git clone https://github.com/tanaygattani8/Amazon_Review_Sentiment_Analysis.git
cd Amazon_Review_Sentiment_Analysis
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
jupyter notebook Sentiment_Analysis.ipynb
```

## Key Results

- **Data Reduction:** Cleaned 10,000 noisy records into 919 high-quality labeled samples.
- **Strong Classification Performance:** Achieved high accuracy on the binary sentiment task using TF-IDF features.
- **Key Findings:** Sentiment-bearing words like "excellent", "broken", "love", and "terrible" are top discriminating features across both classes.

## Author

**Tanay Gattani** - [@tanaygattani8](https://github.com/tanaygattani8)