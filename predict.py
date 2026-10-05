import os
import json
import argparse

import torch
from transformers import Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm

from data_loader import get_dataloader, load_manifest
from model import MERTFineTune


SAMPLE_RATE = 24000

DATASETS = {
    "A": {
        "folder": "dataset_A",
        "classes": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    },
    "B": {
        "folder": "dataset_B",
        "classes": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"],
    },
}

# Checkpoints are stored directly in the project directory.
CHECKPOINT_FILES = {
    "A": "checkpoint_A.pt",
    "B": "checkpoint_B.pt",
}


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["A", "B", "A+B"], default="both")
parser.add_argument("--dataset-dir", required=True)
parser.add_argument("--output", default=None)
args = parser.parse_args()


project_dir = os.path.dirname(os.path.abspath(__file__))
output_path = args.output or os.path.join(project_dir, f"predictions_{args.dataset}.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_datasets = (["A", "B"] if args.dataset == "both" else [args.dataset])


def prepare_batch(chunks, processor):
    batch_size, num_chunks, num_samples = chunks.shape

    waveforms = chunks.reshape(batch_size * num_chunks, num_samples).cpu().numpy()

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


predictions = {}

for dataset_key in selected_datasets:
    dataset_config = DATASETS[dataset_key]
    classes = dataset_config["classes"]

    dataset_dir = os.path.join(args.dataset_dir, dataset_config["folder"])
    manifest_path = os.path.join(dataset_dir, "manifest.csv")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load the fixed checkpoint from the project directory.
    checkpoint_path = os.path.join(project_dir, "checkpoints", CHECKPOINT_FILES[dataset_key])

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_name = checkpoint.get("model_name")

    checkpoint_classes = checkpoint.get("classes")

    print(f"Dataset: {dataset_key}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Encoder: {model_name}")

    data = load_manifest(manifest_path, classes=classes)
    test_loader = get_dataloader(
        data,
        split="test",
        batch_size=1,
        num_workers=0,
    )

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = MERTFineTune(model_name=model_name, n_classes=len(classes)).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset_predictions = {}

    with torch.inference_mode():
        for chunks, _labels, sample_ids in tqdm(test_loader, desc=f"Predicting Dataset {dataset_key}"):
            input_values, attention_mask, batch_size, num_chunks = (prepare_batch(chunks, processor))

            logits = model(
                input_values=input_values,
                batch_size=batch_size,
                num_chunks=num_chunks,
                attention_mask=attention_mask,
            )

            # Get the three highest-scoring classes, in descending score order.
            top3_indices = logits.topk(k=3, dim=1).indices.cpu().tolist()

            for sample_id, indices in zip(sample_ids, top3_indices):
                dataset_predictions[sample_id] = [classes[index] for index in indices]

    predictions[dataset_config["folder"]] = dataset_predictions
    print(
        f"Generated {len(dataset_predictions)} predictions "
        f"for Dataset {dataset_key}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


output_dir = os.path.dirname(os.path.abspath(output_path))
os.makedirs(output_dir, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(predictions, file, ensure_ascii=False, indent=4)

print("Saved predictions:", os.path.abspath(output_path))