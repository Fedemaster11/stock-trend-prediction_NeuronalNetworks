

# Stock Trend Prediction with Neural Networks

*Author:* Federico David Macias Orozco  
*Matriculation:* 4730600  
*University:* Heidelberg University  
*Course:* Neuronal Networks — Sommer Semester 2025  
*Supervisor:* Prof. Michael Staniek  

---

## Project Summary

This project tests whether *sequence models* (LSTM, BiLSTM, 1D-CNN) can predict short-term stock direction using engineered *price features* and coarse *tweet sentiment. We benchmark against simple baselines (random, constant) and classical ML (Logistic Regression, Random Forest), and we evaluate on a **margin-band label* to reduce noise.

Key takeaway: classical baselines hover around random; sequence models achieve *F1 up to ~0.76* with very high recall on some tickers, but *ROC AUC ~0.55–0.61* indicates weak probability ranking—markets are noisy, and daily VADER sentiment adds little signal.

---

## Data & Labels

- *Tickers:* MSFT, GOOGL, TSLA. (PLTR, RHM.DE dropped due to sparse coverage.)
- *Features:* OHLCV + engineered indicators (log returns, MA(5), MA(20), 20-day volatility).  
  Sentiment (VADER) daily aggregates: mean_sent, pos_share, neg_share, n_tweets (only meaningful for MSFT post-2021).
- *Sequences:* 60 trading days → predict *10-day* horizon.
- *Labels (margin-band):*

  \[
  r_{t \rightarrow t+h}=\frac{\text{AdjClose}{t+h}}{\text{AdjClose}{t}}-1,\quad
  y_t=
  \begin{cases}
  1 & \text{if } r \ge 0.005\\
  0 & \text{if } r \le -0.005\\
  \text{ignore} & \text{otherwise}
  \end{cases}
  \]

- *Splits:* chronological 70 % train / 15 % val / 15 % test (no leakage).

Dataset sizes (approx.): MSFT ≈ 2.8k, GOOGL ≈ 2.6k, TSLA ≈ 2.1k sequences; total ≈ 7.5k.

---

## Results (short)

- *Baselines:*  
  Random & Constant ≈ chance.  
  Logistic & Random Forest look okay on naive “price-only labels” but *collapse to ≈ random under margin-band*. Sentiment features carried ~0 importance.

- *Deep models (margin-band, test):*
  - *LSTM:* MSFT F1 0.715, TSLA 0.585, GOOGL 0.718 (AUC ≈ 0.46–0.62).  
  - *BiLSTM:* MSFT 0.715, TSLA 0.688, GOOGL 0.756.  
  - *CNN1D:* MSFT 0.715, TSLA 0.597, *GOOGL 0.762 (recall 0.984)*.  
  Overall *AUC ~0.55–0.61*.

How to read this: F1 is reported *at a tuned probability threshold* chosen on the validation set to maximize F1, then applied to test. ROC AUC is threshold-free and shows probability ranking quality; the modest AUC means limited confidence for ranking/trading even when F1 looks good.

---

## Repository Structure

stock-trend-prediction_NeuronalNetworks/
├─ data/
│  ├─ raw/              # yfinance prices, raw tweets
│  ├─ interim/          # tweet_scored.csv, sentiment_daily.csv
│  └─ processed/        # merged & engineered tables, *.npz sequences
├─ models/              # saved sklearn models & scalers (.joblib)
├─ results/             # CSV/PNG summaries for the report
├─ runs/                # deep model checkpoints & logs (large)
├─ scripts/
│  ├─ build_features.py
│  ├─ build_sentiment_daily.py
│  ├─ merge_sentiment_script.py
│  ├─ fe_phase2_for_schema.py
│  ├─ make_sequences_phase3.py
│  ├─ train_lstm_phase3.py
│  ├─ train_bilstm_phase3.py
│  ├─ train_cnn1d_phase3.py
│  ├─ eval_threshold_sweep.py
│  ├─ collect_and_plot_results.py
│  └─ … (diagnostics & checks)
├─ baseline_logreg.py
├─ baseline_rf.py
├─ requirements.txt
├─ README.md
└─ report/ (LaTeX + final PDF)

---

## Environment Setup

Windows PowerShell (repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Quick sanity check:

python .\test_torch.py


⸻

Reproduce the Pipeline

1) Build features and sentiment

python .\scripts\build_features.py
python .\scripts\build_sentiment_daily.py
python .\scripts\merge_sentiment_script.py
python .\scripts\fe_phase2_for_schema.py

Outputs: data/processed/merged_dataset_with_sent.csv and features_enriched_schema.parquet.

2) Make sequence datasets

python .\scripts\make_sequences_phase3.py --input .\data\processed\features_enriched_schema.parquet --horizon 10 --seq-len 60 --out .\data\processed\seq_MSFT_h10_L60_MB.npz --ticker MSFT
python .\scripts\make_sequences_phase3.py --input .\data\processed\features_enriched_schema.parquet --horizon 10 --seq-len 60 --out .\data\processed\seq_TSLA_h10_L60_MB.npz --ticker TSLA
python .\scripts\make_sequences_phase3.py --input .\data\processed\features_enriched_schema.parquet --horizon 10 --seq-len 60 --out .\data\processed\seq_GOOGL_h10_L60_MB.npz --ticker GOOGL

3) Train deep models

python .\scripts\train_lstm_phase3.py --npz .\data\processed\seq_MSFT_h10_L60_MB.npz --hidden 64 --layers 1 --epochs 40 --batch 128
python .\scripts\train_bilstm_phase3.py --npz .\data\processed\seq_TSLA_h10_L60_MB.npz --hidden 64 --layers 1 --epochs 40 --batch 128 --bidir
python .\scripts\train_cnn1d_phase3.py --npz .\data\processed\seq_GOOGL_h10_L60_MB.npz --epochs 40 --batch 128

Each trainer saves a run under runs/<model>_YYYYMMDD_HHMMSS/ with checkpoints and metric logs.

4) Threshold sweep & evaluation

python .\scripts\eval_threshold_sweep.py --npz .\data\processed\seq_MSFT_h10_L60_MB.npz --run-dir .\runs\bilstm_20250930_161619 --metric f1

Repeat for each ticker/model run.

⸻

Baselines

python .\baseline_logreg.py
python .\baseline_rf.py

Baselines save .joblib models under models/ and CSV summaries under results/.

⸻

How to Read the Results
	•	ROC AUC (~0.55–0.61) captures ranking quality (random = 0.50).
	•	F1, Precision, Recall are computed after threshold tuning (on validation).
	•	High recall but low precision (e.g., MSFT) = catch most upward moves but with false alarms.
	•	Deep models significantly outperform baselines under margin-band labels.

⸻

What to Commit vs Ignore

Commit:
	•	scripts/ (all Python pipeline scripts)
	•	baseline_*.py
	•	requirements.txt
	•	README.md
	•	results/ small CSVs/plots
	•	report/ (LaTeX + final PDF)

Ignore (.gitignore):

.venv/
venv/
_pycache_/
data/raw/
data/interim/
data/processed/*.npz
data/processed/*.parquet
models/*.joblib
runs/**
*.pt


⸻

Minimal Reproduction
	1.	Build features → sequences.
	2.	Train LSTM, BiLSTM, CNN1D on each ticker.
	3.	Threshold sweep for tuned F1.
	4.	Collect plots:

python .\scripts\collect_and_plot_results.py --runs .\runs --out-dir .\results


⸻

Citation

If you reference this work:

Macias Orozco, F. D. (2025). Stock Trend Prediction with Neural Networks (Neuronal Networks, Summer Semester 2025, Heidelberg University).

⸻


MIT or another license of your choice.

---
