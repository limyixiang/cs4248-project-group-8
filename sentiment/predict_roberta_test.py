import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoBERTa sentiment inference on a CSV test set.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to the input test CSV.",
    )
    parser.add_argument(
        "--input-dataset-dir",
        type=Path,
        default=BASE_DIR / "dataset" / "preprocessed_dataset",
        help="Path to a dataset saved by datasets.save_to_disk().",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Split name to use when --input-dataset-dir is provided.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=BASE_DIR / "models" / "roberta_base",
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "results" / "roberta_base_test_predictions.csv",
        help="Path to write predictions CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def read_test_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding="cp1252")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8")


def read_test_split_from_dataset(dataset_dir: Path, split: str) -> pd.DataFrame:
    dataset_obj = load_from_disk(str(dataset_dir))

    if isinstance(dataset_obj, DatasetDict):
        if split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Split '{split}' not found in dataset. Available: {available}")
        ds = dataset_obj[split]
    else:
        # If a single Dataset is loaded, ignore split and use it directly.
        ds = dataset_obj

    return ds.to_pandas()


def build_label_name_map(model) -> dict:
    id2label = getattr(model.config, "id2label", {}) or {}
    normalized = {int(k): str(v) for k, v in id2label.items()}

    # Map generic LABEL_n names to the project sentiment labels.
    if normalized and all(v.startswith("LABEL_") for v in normalized.values()):
        return {idx: DEFAULT_LABEL_MAP.get(idx, f"label_{idx}") for idx in sorted(normalized.keys())}

    if normalized:
        return {idx: name.lower() for idx, name in normalized.items()}

    return DEFAULT_LABEL_MAP.copy()


def predict_sentiment(df: pd.DataFrame, model_dir: Path, batch_size: int) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    label_map = build_label_name_map(model)
    texts = df["text"].fillna("").astype(str).tolist()

    predictions = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(preds)

    result = pd.DataFrame()
    if "textID" in df.columns:
        result["textID"] = df["textID"]

    result["predicted_label_id"] = predictions
    result["predicted_sentiment"] = [label_map.get(i, f"label_{i}") for i in predictions]

    if "text" in df.columns:
        result["text"] = df["text"]

    return result


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv.resolve() if args.input_csv else None
    input_dataset_dir = args.input_dataset_dir.resolve() if args.input_dataset_dir else None
    model_dir = args.model_dir.resolve()
    output_csv = args.output_csv.resolve()

    if input_csv is not None and not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if input_csv is None:
        if input_dataset_dir is None or not input_dataset_dir.exists():
            raise FileNotFoundError(f"Input dataset directory not found: {input_dataset_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if input_csv is not None:
        df = read_test_csv(input_csv)
        input_desc = str(input_csv)
    else:
        df = read_test_split_from_dataset(input_dataset_dir, args.dataset_split)
        input_desc = f"{input_dataset_dir} (split='{args.dataset_split}')"

    pred_df = predict_sentiment(df, model_dir, args.batch_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)

    print(f"Input source: {input_desc}")
    print(f"Loaded rows: {len(df)}")
    print(f"Saved predictions: {len(pred_df)}")
    print(f"Output file: {output_csv}")


if __name__ == "__main__":
    main()
