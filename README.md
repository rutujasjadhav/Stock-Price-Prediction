# Stock Price Movement Prediction Using Technical Indicators

---

## Overview

This project predicts next-day stock price movement for 9 stocks across 6 market sectors using 11 technical indicators derived from historical OHLCV data. The problem is framed as both a classification task (up/down direction) and a regression task (return magnitude). All models are evaluated using walk-forward cross-validation to prevent data leakage.

**Stocks:** AAPL, MSFT, JPM, GS, JNJ, PFE, SPOT, AMZN, TSLA  
**Period:** January 2021 – April 2026  
**Features:** SMA (5/10/20), EMA (5/10/20), RSI-14, MACD, MACD Signal, Volatility, Volume Change  
**Classification models:** Logistic Regression, SVM (RBF), Feedforward DNN  
**Regression models:** Linear Regression + Polynomial Features, Random Forest, Feedforward DNN  
**Stretch goal:** FinBERT-based financial news sentiment analysis

---

## Results Summary

| Task | Best Model | Key Metric |
|---|---|---|
| Classification | SVM (RBF) — Accuracy / LR — F1 | Accuracy 0.5039, F1 0.4896 |
| Regression | Random Forest | RMSE 0.019402, MAE 0.013967 |

---

## Project Structure

```
Stock-Price-Prediction/
├── 01_data_collection.ipynb
├── 02_feature_engineering.ipynb
├── 03_validation_setup.ipynb
├── 04_classification.ipynb
├── 05_regression.ipynb
├── 06_sentiment.ipynb
├── 07_results.ipynb
├── README.md
└── data/                        # created automatically when notebooks run
    ├── raw/                     # OHLCV CSVs per ticker
    ├── features/                # engineered feature CSVs per ticker
    ├── validation/              # walk-forward split plot
    ├── sentiment/               # sentiment score CSVs per ticker
    └── results/                 # model result CSVs and plots
```

---

## Setup

### Requirements

Python 3.10+ is required. Install all dependencies with:

```bash
pip install yfinance pandas numpy scikit-learn torch transformers matplotlib seaborn jupyter ipykernel
```

### Clone the repository

```bash
git clone https://github.com/<your-username>/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

---

## How to Run

Open the project folder in VS Code or Jupyter and run each notebook from top to bottom in order. All `data/` subfolders are created automatically — no manual setup needed.

| Notebook | What it does | Approx. runtime |
|---|---|---|
| 01_data_collection | Downloads OHLCV data via yfinance, saves to `data/raw/` | ~10 sec |
| 02_feature_engineering | Computes 11 technical indicators, normalizes, saves to `data/features/` | ~20 sec |
| 03_validation_setup | Defines walk-forward splits, saves validation plot | ~5 sec |
| 04_classification | Trains LR, SVM, DNN classifiers, saves results and plots | ~30 sec |
| 05_regression | Trains Linear Reg, RF, DNN regressors, saves results and plots | ~1 min |
| 06_sentiment | FinBERT sentiment pipeline, with vs without comparison | ~45 sec |
| 07_results | Loads all saved results, generates final summary plots and analysis | ~5 sec |

### Notes

- All paths use `BASE_DIR = "."` so notebooks run correctly from any machine as long as they are opened from inside the project folder.
- Notebook 06 downloads FinBERT from HuggingFace on first run (~500MB, cached afterwards). An internet connection is required.
- Sentiment coverage is limited to recent articles only (~10 per ticker from yfinance). Historical coverage requires a paid news API such as NewsAPI or Alpha Vantage.

---

## Reproducing Results

Run notebooks 01–07 in order. Results are saved automatically to `data/results/` as CSV files after each model notebook runs. Notebook 07 reads these CSVs and generates all final plots — no hardcoded numbers anywhere.
