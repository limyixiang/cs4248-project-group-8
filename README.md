# CS4248 Project Group 8

Twitter sentiment analysis and clustering pipeline for CS4248, combining:

- sentiment-aware preprocessing,
- classical and transformer-based sentiment modeling,
- unsupervised clustering for topic/structure discovery.

## Repository Overview

```text
.
├── requirements.txt
├── dataset/
│   ├── train.csv
│   └── test.csv
├── preprocessing/
│   ├── dataset_preprocessing.ipynb
│   ├── feature_analysis.ipynb
│   └── preprocessing.py
├── clustering/
│   ├── agglomerative.ipynb
│   ├── cluster_analysis.ipynb
│   ├── hdbscan.ipynb
│   ├── kmeans.ipynb
│   ├── twitter_hdbscan_cluster_labels.csv
│   ├── twitter_hdbscan_cluster_summary.csv
│   ├── twitter_hdbscan_clustered.csv
│   ├── twitter_hdbscan_param_search.csv
│   ├── twitter_kmeans_cluster_labels.csv
│   ├── twitter_kmeans_cluster_summary.csv
│   └── twitter_kmeans_clustered.csv
└── sentiment/
    ├── classical_baselines.ipynb
    ├── clustered_sentiment_analysis.ipynb
    ├── sanity_check.ipynb
    ├── roberta_base.py
    ├── predict_roberta_test.py
    ├── plots/
    ├── results/
    │   └── roberta_base_test_predictions.csv
    └── training_logs/
```

## Dataset

Primary data files are in `dataset/`:

- `train.csv`: labeled training data.
- `test.csv`: held-out test data.

Observed schema:

- `train.csv`: `textID`, `text`, `selected_text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`, `Population -2020`, `Land Area (Km^2)`, `Density (P/Km^2)`
- `test.csv`: `textID`, `text`, `sentiment`, `Time of Tweet`, `Age of User`, `Country`, `Population -2020`, `Land Area (Km^2)`, `Density (P/Km^2)`

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources (automatic fallback)

`preprocessing/preprocessing.py` auto-downloads:

- `stopwords`
- `wordnet`
- `omw-1.4`

No manual action is needed unless your environment blocks downloads.

## Workflow

Typical end-to-end order:

1. Run preprocessing notebooks/functions.
2. Build processed Hugging Face datasets for downstream tasks.
3. Run clustering notebooks (KMeans/HDBSCAN/Agglomerative).
4. Train/evaluate sentiment models (classical baselines and RoBERTa).
5. Analyze cluster-level sentiment behavior.

## Preprocessing

### `preprocessing/preprocessing.py`

Provides two cleaning strategies:

- `clean_text(text)`: aggressive cleaning for classical ML.
- `light_clean(text)`: minimal normalization for transformer models.
- `preprocess_df(df, text_col="text")`: adds both `clean_text` and `light_text` columns.

Example:

```python
import pandas as pd
from preprocessing.preprocessing import preprocess_df

df = pd.read_csv("dataset/train.csv", encoding="cp1252")
processed = preprocess_df(df, text_col="text")
print(processed[["text", "clean_text", "light_text"]].head())
```

### `preprocessing/dataset_preprocessing.ipynb`

Notebook for building saved Hugging Face datasets used by clustering and sentiment experiments.

The notebook contains cells that save variants such as:

- `dataset/preprocessed_dataset`
- `dataset/mini_dataset`
- `dataset/text_only`
- `dataset/full_text_and_sentiment`

Run this notebook before running notebooks/scripts that call `load_from_disk(...)`.

## Clustering

Notebooks in `clustering/`:

- `kmeans.ipynb`
- `hdbscan.ipynb`
- `agglomerative.ipynb`
- `cluster_analysis.ipynb`

Each clustering notebook loads a local dataset path (configured near the top via `DATASET_NAME` and `DATASET_SPLIT`) and performs preprocessing/embedding/clustering experiments.

Current clustering result files include:

- `twitter_kmeans_clustered.csv`
- `twitter_kmeans_cluster_summary.csv`
- `twitter_kmeans_cluster_labels.csv`
- `twitter_hdbscan_clustered.csv`
- `twitter_hdbscan_cluster_summary.csv`
- `twitter_hdbscan_cluster_labels.csv`
- `twitter_hdbscan_param_search.csv`

## Sentiment Modeling

### Classical baselines

Use `sentiment/classical_baselines.ipynb` for traditional ML baselines.

### RoBERTa fine-tuning

`sentiment/roberta_base.py` fine-tunes `FacebookAI/roberta-base` using a saved dataset loaded from:

- `dataset/preprocessed_dataset` (default in script)

Run from repository root:

```bash
python sentiment/roberta_base.py
```

Expected outputs (created during/after training):

- `models/roberta_base` (saved model)
- `results` and `logs` directories (trainer artifacts)
- confusion matrix image in `plots/` (timestamped)

Notes:

- Script requires CUDA and raises an error when GPU is unavailable.
- You may need to edit training arguments directly in the script for hyperparameter changes.

### Test inference with fine-tuned model

Use explicit paths to avoid path ambiguity:

```bash
python sentiment/predict_roberta_test.py \
  --input-csv dataset/test.csv \
  --model-dir models/roberta_base \
  --output-csv sentiment/results/roberta_base_test_predictions.csv
```

Output columns:

- `textID` (if available)
- `predicted_label_id`
- `predicted_sentiment`
- `text`

## Additional Notebooks

- `preprocessing/feature_analysis.ipynb`: analysis behind preprocessing choices.
- `sentiment/sanity_check.ipynb`: spot checks/validation.
- `sentiment/clustered_sentiment_analysis.ipynb`: sentiment behavior by cluster.

## Reproducibility Tips

- Always run commands from repository root unless a notebook explicitly assumes another working directory.
- Prefer absolute paths in notebook config cells (`DATASET_NAME`) to avoid OS/path issues.
- If text decoding issues occur, use `encoding="cp1252"` for this dataset.

## Troubleshooting

- `FileNotFoundError` for dataset/model paths:
  - Confirm preprocessing artifacts exist under `dataset/`.
  - Pass explicit CLI paths to `predict_roberta_test.py`.
- NLTK resource errors:
  - Ensure internet access for first-run corpus downloads.
- CUDA errors in `roberta_base.py`:
  - Use a GPU-enabled environment or adjust script logic to support CPU.

## License

No project license file is currently included in the repository.

## AI Declaration

This README.md file was generated using GPT-5.3-Codex.