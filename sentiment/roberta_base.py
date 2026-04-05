# Import libraries
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoConfig
from datasets import load_dataset, DatasetDict, load_from_disk
from collections import Counter

# Load the TSAD dataset
dataset_dir = "dataset/preprocessed_dataset"
dataset = load_from_disk(dataset_dir)

print(dataset["train"])

# Load the BERT model
pretrained_model = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# Tokenize the text field in each batch
def preprocess_function(examples):
    return tokenizer(list(examples["text"]), padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets.set_format("torch")

if dataset_dir == "dataset/preprocessed_dataset":
    # Split train into train/validation, keep original test split
    train_val_split = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = tokenized_datasets["test"]
else:
    # Use this if using the mini dataset
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["val"]
    test_dataset = tokenized_datasets["test"]

train_dataset = train_dataset.rename_column("sentiment", "labels")
val_dataset = val_dataset.rename_column("sentiment", "labels")
test_dataset = test_dataset.rename_column("sentiment", "labels")

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Check label distribution in the train dataset
labels = train_dataset["labels"]
print("Label distribution in train dataset:", Counter(int(x.item()) if torch.is_tensor(x) else int(x) for x in labels))

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Define training arguments (epoch)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="recall",  # Specify the metric to monitor
    greater_is_better=True       # Specify if higher values of the metric are better
)

# Infer class count from the prepared training split
num_labels = len(set(int(x.item()) if torch.is_tensor(x) else int(x) for x in train_dataset["labels"]))

config = AutoConfig.from_pretrained(
    pretrained_model,
    num_labels=num_labels,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
 )
model = RobertaForSequenceClassification.from_pretrained(pretrained_model, config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# Force GPU training for this run.
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this notebook kernel. Restart the kernel and rerun from Cell 1.")
device = torch.device("cuda")
model = model.to(device)

training_args.use_cpu = False
training_args.fp16 = True
training_args.num_train_epochs = 5
training_args.learning_rate = 2e-5

print("Model is on:", next(model.parameters()).device)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Learning rate:", training_args.learning_rate)
print("num_labels:", num_labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

trainer.evaluate()

trainer.save_model("./models/roberta_base")

# Conduct testing on the test dataset
test_results = trainer.predict(test_dataset)

# Extract predictions and metrics
predictions = test_results.predictions.argmax(-1)  # Convert logits to class predictions
metrics = test_results.metrics  # Contains accuracy, F1, precision, recall, etc.

# Build and save confusion matrix plot with a timestamped filename.
y_true = [int(x.item()) if torch.is_tensor(x) else int(x) for x in test_dataset["labels"]]
y_pred = [int(x) for x in predictions]
label_order = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=label_order)

os.makedirs("./plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"./plots/confusion_matrix_{timestamp}.png"

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_order,
    yticklabels=label_order,
)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to: {plot_path}")

# Print metrics
print("Test Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")