# Stock Price Movement Prediction Using Technical Indicators

**Authors:** Rutuja Jadhav · Bharani Raaj Thiagarajan

---

## Overview

This project predicts next-day stock price movement for 9 stocks across 6 market sectors using 11 technical indicators derived from historical OHLCV data. The problem is framed as both a classification task (up/down direction) and a regression task (return magnitude). All models are evaluated using walk-forward cross-validation to prevent data leakage.

- **Stocks:** AAPL, MSFT, JPM, GS, JNJ, PFE, SPOT, AMZN, TSLA
- **Period:** January 2021 – April 2026
- **Features:** SMA (5/10/20), EMA (5/10/20), RSI-14, MACD, MACD Signal, Volatility, Volume Change
- **Classification models:** Logistic Regression, SVM (RBF), Feedforward DNN
- **Regression models:** Linear Regression + Polynomial Features, Random Forest, Feedforward DNN
- **Sentiment Analysis:** FinBERT-based financial news sentiment analysis used as an added feature

---

## Results Summary

### Classification (next-day direction)

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.5034   | 0.5138    | 0.5677 | 0.4896   |
| SVM (RBF)           | 0.5039   | 0.5387    | 0.5096 | 0.4799   |
| DNN (PyTorch)       | 0.4783   | 0.3884    | 0.2307 | 0.2119   |

### Regression (next-day return magnitude)

| Model               | RMSE     | MAE      |
|---------------------|----------|----------|
| Linear Reg + Poly   | 0.028583 | 0.020705 |
| Random Forest       | 0.019402 | 0.013967 |
| DNN (PyTorch)       | 0.021719 | 0.016313 |

**Key finding:** SVM achieved the highest accuracy and Logistic Regression achieved the best F1 on classification. Random Forest is the best regressor. All models hover near 50% accuracy for direction prediction, consistent with the efficient market hypothesis for technical-indicator-only features.

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

## Requirements

- Python 3.9 or higher
- Git
- Internet connection (for data download and FinBERT on first run)

---

## Setup Instructions

### Step 1 — Check Python is installed

**Mac/Linux:**
```bash
python3 --version
```

**Windows:**
```bash
python --version
```

If Python is not installed, download it from https://www.python.org/downloads/  
Make sure to check **"Add Python to PATH"** during installation on Windows.

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/rutujasjadhav/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

---

### Step 3 — Install dependencies

**Mac/Linux:**
```bash
pip3 install yfinance pandas numpy scikit-learn torch transformers matplotlib seaborn jupyterlab ipykernel
```

**Windows:**
```bash
pip install yfinance pandas numpy scikit-learn torch transformers matplotlib seaborn jupyterlab ipykernel
```

This will take 3-5 minutes. PyTorch and Transformers are large packages.

---

### Step 4 — Launch JupyterLab

**Mac/Linux:**
```bash
python3 -m jupyterlab
```

**Windows:**
```bash
python -m jupyterlab
```

Your browser will open automatically showing the project folder.

---

### Step 5 — Run notebooks in order

Open and run each notebook **top to bottom** in this order:

| Notebook | What it does | Approx. runtime |
|----------|-------------|-----------------|
| 01_data_collection | Downloads OHLCV data via yfinance, saves to `data/raw/` | ~10 sec |
| 02_feature_engineering | Computes 11 technical indicators, normalizes, saves to `data/features/` | ~20 sec |
| 03_validation_setup | Defines walk-forward splits, saves validation plot | ~5 sec |
| 04_classification | Trains LR, SVM, DNN classifiers, saves results and plots | ~30 sec |
| 05_regression | Trains Linear Reg, RF, DNN regressors, saves results and plots | ~1 min |
| 06_sentiment | FinBERT sentiment pipeline, with vs without comparison | ~45 sec |
| 07_results | Loads all saved results, generates final summary plots and analysis | ~5 sec |

All `data/` subfolders are created automatically — no manual setup needed.

---

### Notes

- All paths use `BASE_DIR = "."` so notebooks run correctly from any machine as long as they are opened from inside the project folder.
- Notebook 06 downloads FinBERT from HuggingFace on first run (~500MB, cached afterwards). An internet connection is required.
- Sentiment coverage is limited to recent articles only (~10 per ticker from yfinance). Historical coverage requires a paid news API such as NewsAPI or Alpha Vantage.

---
