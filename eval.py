import os
import argparse

import torch
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm

from data_loader import get_dataloader, load_manifest
from model import MERTFineTune


SAMPLE_RATE = 24000

ENCODERS = {
    "mert_v1_95m": "m-a-p/MERT-v1-95M",
    "culturemert_95m": "ntua-slp/CultureMERT-95M",
}

DATASETS = {
    "A": {
        "dataset_dir": "/home/benjaminwu/mir_course_dataset/hw1/dataset_A",
        "classes": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    },
    "B": {
        "dataset_dir": "/home/benjaminwu/mir_course_dataset/hw1/dataset_B",
        "classes": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
parser.add_argument("--dataset-dir", default=None)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

dataset_config = DATASETS[args.dataset]
classes = dataset_config["classes"]
dataset_dir = args.dataset_dir or dataset_config["dataset_dir"]
manifest_path = os.path.join(dataset_dir, "manifest.csv")
model_name = ENCODERS[args.encoder]

project_dir = os.path.dirname(os.path.abspath(__file__))
default_checkpoint = os.path.join(project_dir, "checkpoints", f"finetune_{args.dataset}_{args.encoder}.pt")
checkpoint_path = args.checkpoint or default_checkpoint

data = load_manifest(manifest_path, classes=classes)
validation_loader = get_dataloader(
    data,
    split="validation",
    batch_size=1,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")
checkpoint_classes = checkpoint.get("classes")

processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
model = MERTFineTune(model_name=model_name, n_classes=len(classes)).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def prepare_batch(chunks):
    batch_size, num_chunks, num_samples = chunks.shape

    waveforms = chunks.reshape(batch_size * num_chunks, num_samples,).cpu().numpy()

    inputs = processor(
        list(waveforms),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    return input_values, attention_mask, batch_size, num_chunks


all_true = []
all_pred = []
top3_correct = 0
total = 0

with torch.inference_mode():
    for chunks, labels, sample_ids in tqdm(validation_loader,desc="Evaluating validation", leave=False):
        labels = labels.to(device)

        input_values, attention_mask, batch_size, num_chunks = prepare_batch(chunks)

        logits = model(
            input_values=input_values,
            batch_size=batch_size,
            num_chunks=num_chunks,
            attention_mask=attention_mask,
        )

        top1 = logits.argmax(dim=1)
        top3 = logits.topk(k=3, dim=1).indices

        top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(top1.cpu().tolist())

top1_accuracy = sum(true == pred for true, pred in zip(all_true, all_pred)) / total
top3_accuracy = top3_correct / total

matrix = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(classes))),
)

print("Dataset:", args.dataset)
print("Encoder:", model_name)
print("Checkpoint:", checkpoint_path)
print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
print(f"Validation Top-1: {top1_accuracy:.4f} ({sum(t == p for t, p in zip(all_true, all_pred))}/{total})")
print(f"Validation Top-3: {top3_accuracy:.4f} ({top3_correct}/{total})")
print("Class order:", classes)
print("Confusion matrix (rows=true, columns=predicted):")
print(matrix)

# Confusion Matrix Visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=classes,
)

fig, ax = plt.subplots(figsize=(8, 6))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False,
)
ax.set_title(
    f"Validation Confusion Matrix\n"
    f"{args.dataset} | {args.encoder}"
)
fig.tight_layout()

output_dir = os.path.dirname(checkpoint_path)
image_path = os.path.join(
    output_dir,
    f"confusion_matrix_{args.dataset}_{args.encoder}.png",
)
fig.savefig(image_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved confusion matrix image:", image_path)